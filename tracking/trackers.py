"""
Os três algoritmos de rastreamento, todos com o mesmo jeito de usar:

    tracker = create_tracker("bytetrack", params, fps, width, height)
    tracks = tracker.update(detections, frame)

    detections: lista de caixas [x1, y1, x2, y2, conf] que o YOLO de pessoas achou no frame
    tracks:     dicionário {id: [x1, y1, x2, y2]} com as pessoas rastreadas neste frame

Cada um usa a implementação de referência:
    sort       sort.py original do autor (github.com/abewley/sort), baixado em third_party/ na 1ª vez
    deepsort   pacote deep-sort-realtime
    bytetrack  implementação do Ultralytics
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # raiz do projeto
THIRD_PARTY = ROOT / "third_party"
SORT_URL = "https://raw.githubusercontent.com/abewley/sort/master/sort.py"


def create_tracker(algorithm, params, fps, width, height):
    if algorithm == "sort":
        return SortTracker(params)
    if algorithm == "deepsort":
        return DeepSortTracker(params)
    if algorithm == "bytetrack":
        return ByteTrackTracker(params, fps, width, height)
    raise ValueError(f"Algoritmo desconhecido: {algorithm}")


class SortTracker:
    """SORT (Bewley et al., ICIP 2016): filtro de Kalman + IoU + algoritmo húngaro. Não olha a imagem."""

    def __init__(self, params):
        # Baixa o sort.py original na primeira vez
        if not (THIRD_PARTY / "sort.py").exists():
            print(f"Baixando o sort.py original de {SORT_URL}")
            THIRD_PARTY.mkdir(exist_ok=True)
            urllib.request.urlretrieve(SORT_URL, THIRD_PARTY / "sort.py")
        sys.path.insert(0, str(THIRD_PARTY))

        # O sort.py tenta abrir uma janela do matplotlib (só para a demo dele). Desligo isso antes do import.
        import matplotlib
        matplotlib.use("Agg")
        original_use = matplotlib.use
        matplotlib.use = lambda *args, **kwargs: None
        import sort
        matplotlib.use = original_use

        self.min_confidence = params["min_confidence"]  # o SORT não tem limiar de confiança: o filtro é nosso
        self.sort = sort.Sort(max_age=params["max_age"], min_hits=params["min_hits"],
                              iou_threshold=params["iou_threshold"])

    def update(self, detections, frame):
        boxes = [d for d in detections if d[4] >= self.min_confidence]
        boxes = np.array(boxes, dtype=float) if boxes else np.empty((0, 5))
        output = self.sort.update(boxes)  # cada linha: x1, y1, x2, y2, id
        return {int(row[4]): row[:4] for row in output}


class DeepSortTracker:
    """DeepSORT (Wojke et al., ICIP 2017): SORT + aparência. Recorta cada pessoa da imagem."""

    def __init__(self, params):
        import torch
        from deep_sort_realtime.deepsort_tracker import DeepSort

        use_gpu = torch.cuda.is_available()
        self.min_confidence = params["min_confidence"]
        self.deepsort = DeepSort(
            max_iou_distance=params["max_iou_distance"],
            max_age=params["max_age"],
            n_init=params["n_init"],
            max_cosine_distance=params["max_cosine_distance"],
            nn_budget=params["nn_budget"],
            nms_max_overlap=params["nms_max_overlap"],
            embedder=params["embedder"],
            half=use_gpu,
            bgr=True,
            embedder_gpu=use_gpu,
        )

    def update(self, detections, frame):
        # O deep-sort-realtime quer cada caixa como ([x, y, largura, altura], confiança, classe)
        boxes = []
        for x1, y1, x2, y2, conf in detections:
            if conf >= self.min_confidence and x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        tracks = {}
        for track in self.deepsort.update_tracks(boxes, frame=frame):
            # Mesmo critério do deep_sort_app.py original: rastro confirmado e visto há no máximo 1 frame
            if track.is_confirmed() and track.time_since_update <= 1:
                tracks[int(track.track_id)] = track.to_ltrb()
        return tracks


class ByteTrackTracker:
    """ByteTrack (Zhang et al., ECCV 2022), do Ultralytics. Não olha a imagem."""

    def __init__(self, params, fps, width, height):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        self.bytetrack = BYTETracker(IterableSimpleNamespace(tracker_type="bytetrack", **params), frame_rate=round(fps))
        self.Boxes = Boxes
        self.image_shape = (height, width)

    def update(self, detections, frame):
        # O Ultralytics quer as caixas como x1, y1, x2, y2, conf, classe
        data = np.zeros((len(detections), 6), dtype=np.float32)
        if detections:
            data[:, :5] = np.array(detections, dtype=np.float32)
        output = self.bytetrack.update(self.Boxes(data, self.image_shape), frame)
        # cada linha: x1, y1, x2, y2, id, score, classe, índice
        return {int(row[4]): row[:4] for row in output}
