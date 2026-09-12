"""Configuração central do harness de treino do TCC.

Ponto único de verdade para raízes de dados, geometria dos dados anotados e
perfis de aumento. Nada de caminho absoluto espalhado pelos scripts.

Layout esperado dos dados (exportação CVAT "YOLO 1.1", uma pasta por minuto):

    <raiz da condição>/min<N>/obj_train_data/frame_NNNNNN.png
                                             frame_NNNNNN.txt
                             train.txt, obj.names, obj.data

A numeração dos frames é global e contínua: a pasta ``min<N>`` cobre os frames
``N*1800 .. (N+1)*1800-1`` (1 minuto a 30 fps).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_DIR / "datasets"
RUNS_DIR = PROJECT_DIR / "runs"

# Condições de captura comparadas no TCC. Cada raiz pode ser sobrescrita por
# variável de ambiente para o código não travar num caminho fixo.
CONDITIONS: dict[str, Path] = {
    "ang": Path(os.environ.get("TCC_DATA_ANG", r"D:\tcc\angulo_aberto")),
    "broadcast": Path(os.environ.get("TCC_DATA_BROADCAST", r"D:\tcc\broadcast")),
}

OBJ_SUBDIR = "obj_train_data"
IMG_EXT = ".png"
CVAT_MANIFEST = "train.txt"

FPS = 30
FRAMES_PER_MIN = FPS * 60  # 1800

# Split 70/30 por bloco temporal dentro de cada minuto: os primeiros 1260
# frames vão para treino, os últimos 540 para validação. Determinístico e
# estável entre cenários — o que é val no cenário de 5 min continua val no de 40.
TRAIN_FRAMES = 1260
VAL_FRAMES = FRAMES_PER_MIN - TRAIN_FRAMES

# Uma única classe. O obj.names exportado pelo CVAT traz uma segunda classe
# espúria ("a") que nenhum label usa; ela é ignorada de propósito.
CLASS_NAMES: dict[int, str] = {0: "player"}

# Seção 4.5 do TCC: pesos guardados nestas épocas para avaliação posterior.
CHECKPOINT_EPOCHS = (10, 25, 50)

DEFAULT_CONDITION = "ang"
DEFAULT_STRIDE = 3  # 1 frame a cada 3 => 10 fps
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 4

# Seção 4.5: duas configurações de aumento. "default" deixa tudo no padrão do
# Ultralytics (linha de base); "scale" só amplia a faixa de escala, mantendo
# fliplr/flipud/translate no padrão, como o texto exige.
AUG_PROFILES: dict[str, dict[str, float]] = {
    "default": {},
    "scale": {"scale": 0.9},
    "identidade": {
        "hsv_h": 0.0,   
        "hsv_s": 0.0,
    },
}

# Throughput medido nesta máquina (GTX 1050 Ti, yolo11n, imgsz 640, batch 8):
# 420 imagens de treino em ~17,1 s de época, ou seja ~24 img/s com a GPU como
# gargalo. Cenários grandes demais para caber no cache de RAM passam a depender
# da leitura do disco externo, que o próprio Ultralytics mediu em ~22 MB/s
# (~11 img/s a 2 MB por PNG) — por isso a estimativa é dada como faixa.
IMGS_PER_SEC_GPU = 24.0
IMGS_PER_SEC_IO = 11.0
VAL_OVERHEAD = 1.15  # validação por época, em cima do tempo de treino


def condition_root(condition: str) -> Path:
    """Raiz dos dados de uma condição de captura."""
    try:
        return CONDITIONS[condition]
    except KeyError:
        raise SystemExit(
            f"Condição desconhecida: {condition!r}. Use uma de: {', '.join(CONDITIONS)}"
        ) from None


def minute_dir(condition: str, minute: int) -> Path:
    return condition_root(condition) / f"min{minute}"


def minute_zip(condition: str, minute: int) -> Path:
    return condition_root(condition) / f"min{minute}.zip"


def frame_index_range(minute: int) -> range:
    """Índices globais de frame cobertos pela pasta min<N>."""
    start = minute * FRAMES_PER_MIN
    return range(start, start + FRAMES_PER_MIN)


def frame_stem(index: int) -> str:
    return f"frame_{index:06d}"


def scenario_name(condition: str, minutes: int, stride: int) -> str:
    """Nome da pasta do dataset montado, ex.: ang_05min_s3."""
    return f"{condition}_{minutes:02d}min_s{stride}"


def scenario_dir(condition: str, minutes: int, stride: int) -> Path:
    return DATASETS_DIR / scenario_name(condition, minutes, stride)


def run_name(condition: str, minutes: int, stride: int, aug: str) -> str:
    """Nome do run do Ultralytics, ex.: ang_05min_s3_default."""
    return f"{scenario_name(condition, minutes, stride)}_{aug}"


def estimate_hours(train_images: int, epochs: int) -> tuple[float, float]:
    """Faixa estimada de horas de treino: (limitado pela GPU, limitado pelo disco)."""

    def hours(rate: float) -> float:
        return (train_images / rate) * VAL_OVERHEAD * epochs / 3600.0

    return hours(IMGS_PER_SEC_GPU), hours(IMGS_PER_SEC_IO)


def safe_streams() -> None:
    """Evita UnicodeEncodeError quando a saída é redirecionada para um pipe."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            pass
