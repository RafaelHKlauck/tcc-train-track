"""
Rastreadores de objeto único (SOT) com a mesma interface:

    tracker = create_tracker("csrt")
    tracker.init(frame_bgr, (x, y, w, h))           # caixa inicial (do gabarito)
    found, box, score = tracker.update(frame_bgr)   # box = (x, y, w, h) ou None; score pode ser None

Clássicos (opencv-contrib-python): csrt, kcf, mosse, mil, medianflow, tld
Com rede, pelo OpenCV (precisam dos .onnx do download_models.py): dasiamrpn, nano, vit
Estado da arte, opcional (GPU NVIDIA + repositório do DAM4SAM): dam4sam

Todos usam os parâmetros padrão das implementações oficiais.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import sot_config

# Metadados de cada algoritmo, salvos no run_info.json (útil para a tabela do TCC)
ALGORITHMS = {
    "csrt": {"family": "correlation filter",
             "paper": "Lukezic et al., Discriminative Correlation Filter with Channel and Spatial Reliability, CVPR 2017"},
    "kcf": {"family": "correlation filter",
            "paper": "Henriques et al., High-Speed Tracking with Kernelized Correlation Filters, TPAMI 2015"},
    "mosse": {"family": "correlation filter",
              "paper": "Bolme et al., Visual Object Tracking using Adaptive Correlation Filters, CVPR 2010"},
    "mil": {"family": "online learning (multiple instance learning)",
            "paper": "Babenko, Yang e Belongie, Visual Tracking with Online Multiple Instance Learning, CVPR 2009"},
    "medianflow": {"family": "optical flow with failure detection",
                   "paper": "Kalal, Mikolajczyk e Matas, Forward-Backward Error: Automatic Detection of Tracking Failures, ICPR 2010"},
    "tld": {"family": "long-term (tracking + detection + learning)",
            "paper": "Kalal, Mikolajczyk e Matas, Tracking-Learning-Detection, TPAMI 2012"},
    "dasiamrpn": {"family": "siamese network (distractor-aware)",
                  "paper": "Zhu et al., Distractor-aware Siamese Networks for Visual Object Tracking, ECCV 2018"},
    "nano": {"family": "siamese network (lightweight)",
             "paper": "sem artigo proprio (NanoTrack v2, github.com/HonglinChu/SiamTrackers)"},
    "vit": {"family": "transformer (lightweight, one-stream)",
            "paper": "sem artigo proprio (VitTrack, OpenCV Zoo / GSoC 2023)"},
    "dam4sam": {"family": "memory-based (SAM 2.1 + distractor-aware memory)",
                "paper": "Videnovic, Lukezic e Kristan, A Distractor-Aware Memory for Visual Object Tracking with SAM2, CVPR 2025"},
}

OPENCV_CLASSIC = ["csrt", "kcf", "mosse", "mil", "medianflow", "tld"]
OPENCV_DNN = ["dasiamrpn", "nano", "vit"]

# Arquivos que cada rastreador com rede precisa em sot_config.MODELS_DIR
MODEL_FILES = {
    "dasiamrpn": {"model": "dasiamrpn_model.onnx",
                  "kernel_cls1": "dasiamrpn_kernel_cls1.onnx",
                  "kernel_r1": "dasiamrpn_kernel_r1.onnx"},
    "nano": {"backbone": "nanotrack_backbone_sim_v2.onnx",
             "neckhead": "nanotrack_head_sim_v2.onnx"},
    "vit": {"net": "object_tracking_vittrack_2023sep.onnx"},
}


def _valid_box(box):
    if box is None:
        return None
    x, y, w, h = (float(v) for v in box)
    if not all(np.isfinite([x, y, w, h])) or w <= 1 or h <= 1:
        return None
    return (x, y, w, h)


class OpenCVTracker:
    """Clássicos e com rede do OpenCV. Recria o objeto a cada init (mais seguro que reaproveitar)."""

    def __init__(self, name, models_dir=None):
        self.name = name
        self.models_dir = Path(models_dir or sot_config.MODELS_DIR)
        self._create, self.params = self._factory(name)
        self.tracker = None
        self.has_score = name in OPENCV_DNN

    def _model_path(self, name, key):
        p = self.models_dir / MODEL_FILES[name][key]
        if not p.exists():
            raise FileNotFoundError(
                f"Modelo {p.name} não encontrado em {self.models_dir}. "
                f"Rode antes: python tracking/SOT/download_models.py {name}")
        return str(p)

    def _factory(self, name):
        legacy = getattr(cv2, "legacy", None)
        need_contrib = ("Instale o opencv-contrib-python (ver README da pasta SOT). "
                        "O opencv-python comum não tem CSRT, KCF nem os 'legacy'.")
        if name in ("csrt", "kcf"):
            fn = getattr(cv2, f"Tracker{name.upper()}_create", None)
            if fn is None:
                raise RuntimeError(need_contrib)
            return fn, {}
        if name == "mil":
            return cv2.TrackerMIL_create, {}
        if name in ("mosse", "medianflow", "tld"):
            cls = {"mosse": "TrackerMOSSE_create", "medianflow": "TrackerMedianFlow_create", "tld": "TrackerTLD_create"}[name]
            if legacy is None or not hasattr(legacy, cls):
                raise RuntimeError(need_contrib)
            return getattr(legacy, cls), {}
        if name == "dasiamrpn":
            p = cv2.TrackerDaSiamRPN_Params()
            p.model = self._model_path(name, "model")
            p.kernel_cls1 = self._model_path(name, "kernel_cls1")
            p.kernel_r1 = self._model_path(name, "kernel_r1")
            return (lambda: cv2.TrackerDaSiamRPN_create(p)), {"onnx": list(MODEL_FILES[name].values())}
        if name == "nano":
            p = cv2.TrackerNano_Params()
            p.backbone = self._model_path(name, "backbone")
            p.neckhead = self._model_path(name, "neckhead")
            return (lambda: cv2.TrackerNano_create(p)), {"onnx": list(MODEL_FILES[name].values())}
        if name == "vit":
            if not hasattr(cv2, "TrackerVit_create"):
                raise RuntimeError("TrackerVit precisa do OpenCV 4.9 ou mais novo.")
            p = cv2.TrackerVit_Params()
            p.net = self._model_path(name, "net")
            # O próprio VitTrack devolve found=False quando o score fica abaixo disto (padrão do OpenCV)
            return (lambda: cv2.TrackerVit_create(p)), {"onnx": list(MODEL_FILES[name].values()),
                                                        "tracking_score_threshold": round(float(p.tracking_score_threshold), 3)}
        raise ValueError(f"Algoritmo desconhecido: {name}")

    def init(self, frame, box):
        self.tracker = self._create()
        x, y, w, h = box
        rect = (int(round(x)), int(round(y)), max(2, int(round(w))), max(2, int(round(h))))
        self.tracker.init(frame, rect)  # legacy devolve bool, o novo devolve None

    def update(self, frame):
        try:
            found, box = self.tracker.update(frame)
        except cv2.error as e:  # alguns rastreadores quebram com caixa saindo da imagem
            if not getattr(self, "_warned", False):
                print(f"[{self.name}] Aviso: erro do OpenCV no update ({str(e).strip()[:120]}). "
                      f"Contando como 'não achou' até a próxima inicialização.")
                self._warned = True
            return False, None, None
        score = None
        if self.has_score:
            try:
                score = float(self.tracker.getTrackingScore())
            except (AttributeError, cv2.error):
                score = None
        box = _valid_box(box) if found else None
        return box is not None, box, score


class DAM4SAMTracker:
    """
    DAM4SAM (CVPR 2025) pelo repositório oficial. Precisa de GPU NVIDIA (o código usa cuda:0)
    e roda bem melhor no Linux/WSL2 (o requirements dele pede triton, que não tem versão para Windows).

    O código original guarda a saída de TODOS os frames na GPU, o que estoura a memória num vídeo de
    20 min. Aqui a memória antiga é podada: ficam só os frames recentes e os que a DRM ainda pode usar
    (o frame inicial e as últimas entradas). Os frames que o modelo de fato consulta continuam iguais.
    """

    def __init__(self, name="dam4sam", repo=None, model=None, keep_recent=64):
        self.name = name
        repo = repo or sot_config.DAM4SAM_REPO
        if not repo or not Path(repo).exists():
            raise RuntimeError("Defina DAM4SAM_REPO em tracking/SOT/sot_config.py (ou --dam4sam-repo) "
                               "com a pasta do repositório clonado do DAM4SAM.")
        sys.path.insert(0, str(Path(repo).resolve()))
        from dam4sam_tracker import DAM4SAMTracker as _Tracker  # noqa: E402
        from PIL import Image  # noqa: E402
        self._Image = Image
        self.model = model or sot_config.DAM4SAM_MODEL
        self.tracker = _Tracker(self.model)
        self.keep_recent = keep_recent
        self.has_score = False
        self.params = {"model": self.model, "repo": str(repo), "keep_recent_frames": keep_recent}
        self._prune_ok = True

    def _pil(self, frame):
        return self._Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _mask_to_box(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        return _valid_box((x1, y1, x2 - x1, y2 - y1))

    def _prune(self):
        """Remove saídas antigas da memória mantendo o estado interno coerente (o SAM2 confere isso)."""
        if not self._prune_ok:
            return
        try:
            st = self.tracker.inference_state
            limit = self.tracker.frame_index - self.keep_recent
            if limit <= 0:
                return
            drm = st["adds_in_drm_per_obj"].get(0, [])
            keep_cond = {0, *sorted(drm)[-8:]}  # o modelo usa no máximo 4 frames da DRM, sempre os mais próximos
            drop_cond = [t for t in drm if t < limit and t not in keep_cond]
            dicts = [st["output_dict"], *st["output_dict_per_obj"].values()]
            for d in dicts:
                for t in [t for t in d["non_cond_frame_outputs"] if t < limit]:
                    del d["non_cond_frame_outputs"][t]
                for t in drop_cond:
                    d["cond_frame_outputs"].pop(t, None)
            if drop_cond:
                st["consolidated_frame_inds"]["cond_frame_outputs"].difference_update(drop_cond)
                st["adds_in_drm_per_obj"][0] = [t for t in drm if t not in set(drop_cond)]
            for t in [t for t in st["frames_already_tracked"] if t < limit]:
                del st["frames_already_tracked"][t]
        except Exception as e:  # se o repositório mudar por dentro, segue sem podar
            print(f"[dam4sam] Aviso: não consegui podar a memória ({e}). Seguindo sem podar.")
            self._prune_ok = False

    def init(self, frame, box):
        x, y, w, h = box
        self.tracker.initialize(self._pil(frame), None, bbox=[float(x), float(y), float(w), float(h)])

    def update(self, frame):
        out = self.tracker.track(self._pil(frame))
        self._prune()
        box = self._mask_to_box(out["pred_mask"])
        return box is not None, box, None


def create_tracker(name, models_dir=None, dam4sam_repo=None):
    name = name.lower()
    if name not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconhecido: {name}. Opções: {', '.join(ALGORITHMS)}")
    if name == "dam4sam":
        return DAM4SAMTracker(repo=dam4sam_repo)
    return OpenCVTracker(name, models_dir)
