"""
Caminhos usados pelos scripts de SOT.

VIDEO e GT de cada captura são os MESMOS arquivos que o tracking/run.py usa:
  video -> VIDEOS, em tracking/run.py
  gt    -> <output>/gt.txt de CAPTURES, em tracking/build_ground_truth.py
           (o gabarito unificado, MOT com o frame começando em 1)
Também dá para passar na linha de comando: --video e --gt.

O gabarito pode estar em dois formatos (o script detecta sozinho):
  - MOT (.txt): frame,id,x,y,w,h,... com frame começando em 1 (export do CVAT / unificado)
  - CSV com cabeçalho: frame, x1, y1, x2, y2 (e opcionalmente visible/visivel)
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAPTURES = {
    "ang": {
        "video": PROJECT_ROOT / "angulo_aberto.mp4",
        "gt": Path(r"D:\tcc\ang_unified\gt.txt"),
    },
    "broadcast": {
        "video": PROJECT_ROOT / "broadcast.mp4",
        "gt": Path(r"D:\tcc\broadcast_unified\gt.txt"),
    },
}

# Detecções de pessoas do YOLO pronto, gravadas pelo tracking/run.py.
# O tracking/evaluate.py usa isso para separar troca de deriva quando a pasta da
# rodada não tem tracks.txt (caso das rodadas de SOT).
def people_detections(capture):
    return PROJECT_ROOT / "results" / capture / "detections" / "people.txt"

# Saídas em results/<captura>/sot/<algoritmo>_<protocolo>/ (ao lado de mot_custom/)
RESULTS_DIR = PROJECT_ROOT / "results"

# Modelos ONNX dos rastreadores com rede do OpenCV (baixados pelo download_models.py)
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Repositório clonado do DAM4SAM (opcional; só para --algorithm dam4sam)
# ex.: r"C:\Users\Rafael Hansen Klauck\codes\DAM4SAM"  ou, no WSL, "/home/rafael/DAM4SAM"
DAM4SAM_REPO = None
DAM4SAM_MODEL = "sam21pp-L"  # "sam21pp-L" (o do artigo) ou "sam21pp-B" (mais leve)
