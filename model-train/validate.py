#!/usr/bin/env python3
"""Valida os detectores treinados nos dados do segundo tempo. Não treina nada.

Uso:
    python validate.py                          # os 2 melhores modelos, 20 min, 10 fps
    python validate.py --cond ang               # só o melhor modelo do ângulo aberto
    python validate.py --cond broadcast --all-runs    # todas as runs de broadcast
    python validate.py --runs ang_10min_s3_default broadcast_40min_s6_default
    python validate.py --all-runs               # best.pt de todas as runs em runs/detect
    python validate.py --weights best epoch025  # escolhe os pesos de cada run
    python validate.py --minutes 16 --allow-partial   # enquanto a anotação não termina
    python validate.py --dry-run                # só confere os dados e mostra o plano

Os dados do segundo tempo seguem o mesmo layout da exportação do CVAT usada no
treino (uma pasta ``min<N>`` por minuto de vídeo), em raízes separadas:

    D:\\tcc\\angulo_aberto_segundo_tempo\\min1 .. min20     (30 fps, 1.800 frames/min)
    D:\\tcc\\broadcast_segundo_tempo\\min1 .. min20         (60 fps, 3.600 frames/min)

Cada modelo é avaliado só no segundo tempo da condição em que foi treinado. Todo
o segundo tempo é teste: não há split, e o stride padrão deixa a amostragem em
10 fps nas duas condições (3 no ângulo aberto, 6 no broadcast), igual ao treino.
Com 20 min isso dá 12.000 imagens em cada condição.

O conjunto de teste é montado com symlinks em
``datasets/segundo_tempo/<condição>_<NN>min_s<stride>/`` e os resultados vão
para ``runs/val/<run>__<peso>/`` e para a tabela ``runs/val/segundo_tempo.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import config
import prepare

# Os dois melhores modelos na validação do treino (maior mAP50-95 em cada condição).
DEFAULT_RUNS = ("ang_20min_s3_identidade", "broadcast_40min_s6_identidade")
DEFAULT_WEIGHTS = ("best",)
DEFAULT_MINUTES = 20

# Mesmo batch que o Ultralytics usou na validação durante o treino (2x o de treino).
DEFAULT_VAL_BATCH = config.DEFAULT_BATCH * 2

TEST_DATASETS_DIR = config.DATASETS_DIR / "segundo_tempo"
VAL_RUNS_DIR = config.RUNS_DIR / "val"
SUMMARY_CSV = VAL_RUNS_DIR / "segundo_tempo.csv"

# Raízes do segundo tempo. O fps vem da condição de treino (config.CONDITIONS):
# é o mesmo vídeo, então 30 fps no ângulo aberto e 60 no broadcast.
TEST_ROOTS: dict[str, Path] = {
    "ang": Path(os.environ.get("TCC_TEST_ANG", r"D:\tcc\angulo_aberto_segundo_tempo")),
    "broadcast": Path(os.environ.get("TCC_TEST_BROADCAST", r"D:\tcc\broadcast_segundo_tempo")),
}

CONDITION_LABELS = {"ang": "ângulo aberto", "broadcast": "broadcast"}


# --------------------------------------------------------------------------- #
# Dados do segundo tempo
# --------------------------------------------------------------------------- #


@dataclass
class TestMinute:
    """O que foi encontrado numa pasta min<N> do segundo tempo."""

    minute: int
    obj_dir: Path
    expected: list[str] = field(default_factory=list)
    usable: set[str] = field(default_factory=set)
    missing_images: int = 0
    missing_labels: int = 0
    zip_present: bool = False
    has_manifest: bool = False
    sampled: int = 0
    empty_labels: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def scan_test_minute(
    condition: str, minute: int, *, strict: bool = False
) -> TestMinute:
    """Inspeciona uma pasta min<N> do segundo tempo sem abrir as imagens."""
    cond = config.get_condition(condition)
    minute_path = TEST_ROOTS[condition] / f"min{minute}"
    obj_dir = minute_path / config.OBJ_SUBDIR
    report = TestMinute(
        minute=minute,
        obj_dir=obj_dir,
        zip_present=(TEST_ROOTS[condition] / f"min{minute}.zip").is_file(),
    )

    if not obj_dir.is_dir():
        what = "pasta não encontrada" if not minute_path.is_dir() else f"sem {config.OBJ_SUBDIR}/"
        msg = f"{what}: {obj_dir if minute_path.is_dir() else minute_path}"
        if report.zip_present:
            msg += f" (min{minute}.zip está lá, falta extrair?)"
        report.errors.append(msg)
        return report

    images: set[str] = set()
    labels: set[str] = set()
    with os.scandir(obj_dir) as entries:
        for entry in entries:
            name = entry.name
            if name.endswith(config.IMG_EXT):
                images.add(name[: -len(config.IMG_EXT)])
            elif name.endswith(".txt"):
                labels.add(name[:-4])

    # O train.txt do CVAT é o manifesto da exportação: diz quais frames a pasta
    # deveria ter, na ordem do vídeo.
    manifest = minute_path / config.CVAT_MANIFEST
    if manifest.is_file():
        stems = {
            Path(line.strip().replace("\\", "/")).stem
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip().lower().endswith(config.IMG_EXT)
        }
        report.expected = sorted(stems)
        report.has_manifest = bool(stems)
    if not report.expected:
        report.expected = sorted(images | labels)
        report.warnings.append(
            f"sem {config.CVAT_MANIFEST}: usando os {len(report.expected)} frames encontrados"
        )

    # Confere o fps: um minuto do vídeo tem fps * 60 frames. Se a pasta tiver
    # outra contagem, o stride deixaria de dar 10 fps efetivos.
    if report.has_manifest and len(report.expected) != cond.frames_per_min:
        report.warnings.append(
            f"{len(report.expected)} frames no {config.CVAT_MANIFEST}, esperado "
            f"{cond.frames_per_min} para {cond.fps} fps — confira o fps desta pasta"
        )

    expected_set = set(report.expected)
    report.missing_images = len(expected_set - images)
    report.missing_labels = len(expected_set - labels)
    report.usable = expected_set & images & labels
    if report.missing_images or report.missing_labels:
        msg = (
            f"{report.missing_images} imagem(ns) e {report.missing_labels} label(s) "
            f"faltando de {len(report.expected)} esperados"
        )
        if report.zip_present:
            msg += f" — min{minute}.zip ainda está lá, extração em andamento?"
        report.errors.append(msg)

    # Checagem de conteúdo por amostragem (ou completa com --strict).
    usable = sorted(report.usable)
    if usable:
        step = 1 if strict else max(1, len(usable) // prepare.SAMPLE_LABELS)
        class_ids: set[str] = set()
        for stem in usable[::step]:
            is_empty, ids = prepare._read_label(obj_dir / f"{stem}.txt")
            report.sampled += 1
            report.empty_labels += int(is_empty)
            class_ids |= ids
        unexpected = class_ids - {str(i) for i in config.CLASS_NAMES}
        if unexpected:
            report.errors.append(
                f"ids de classe inesperados nos labels: {sorted(unexpected)} "
                f"(esperado apenas {sorted(config.CLASS_NAMES)})"
            )

    return report


def select_frames(report: TestMinute, stride: int) -> list[str]:
    """1 frame a cada ``stride``, contado sobre o manifesto do minuto.

    O stride é aplicado na ordem do vídeo (o manifesto completo), e só depois
    ficam de fora os frames que faltam: um frame faltando não desloca a grade
    de amostragem do resto do minuto.
    """
    return [stem for stem in report.expected[::stride] if stem in report.usable]


def check_test_data(
    condition: str, minutes: int, *, strict: bool, allow_partial: bool
) -> list[TestMinute]:
    """Valida min1..minN do segundo tempo e aborta se algo estiver faltando."""
    root = TEST_ROOTS[condition]
    if not root.is_dir():
        raise SystemExit(
            f"Raiz do segundo tempo da condição {condition!r} não existe: {root}\n"
            "Ajuste TEST_ROOTS em validate.py ou a variável de ambiente "
            f"TCC_TEST_{condition.upper()}."
        )

    print(f"\nConferindo {minutes} minuto(s) em {root} ...")
    reports = [scan_test_minute(condition, m, strict=strict) for m in range(1, minutes + 1)]

    for r in reports:
        for w in r.warnings:
            print(f"  [aviso] min{r.minute}: {w}")

    broken = [r for r in reports if not r.ok]
    if broken:
        lines = [f"Faltam dados do segundo tempo ({CONDITION_LABELS[condition]}):"]
        for r in broken:
            lines.extend(f"  - min{r.minute}: {err}" for err in r.errors)
        if allow_partial:
            lines += ["", "--allow-partial: seguindo só com os frames disponíveis."]
            print("\n".join(lines))
        else:
            lines += [
                "",
                "Extraia/anote as pastas que faltam, diminua --minutes, ou use "
                "--allow-partial para validar só com o que existe.",
            ]
            raise SystemExit("\n".join(lines))
    return reports


def build_test_set(
    condition: str,
    minutes: int,
    stride: int,
    *,
    strict: bool,
    allow_partial: bool,
    link_mode: str,
    force: bool,
    dry_run: bool,
) -> tuple[Path, dict]:
    """Monta (ou reaproveita) o conjunto de teste e devolve (data.yaml, manifest)."""
    cond = config.get_condition(condition)
    reports = check_test_data(condition, minutes, strict=strict, allow_partial=allow_partial)

    items: list[tuple[Path, str]] = []
    used_minutes: list[int] = []
    for r in reports:
        chosen = select_frames(r, stride)
        if chosen:
            used_minutes.append(r.minute)
        items.extend((r.obj_dir, stem) for stem in chosen)
    if not items:
        raise SystemExit(f"Nenhum frame selecionado no segundo tempo de {condition!r}.")

    name = f"{condition}_{minutes:02d}min_s{stride}"
    scenario = TEST_DATASETS_DIR / name
    data_yaml = scenario / "data.yaml"
    manifest = {
        "condition": condition,
        "data_root": str(TEST_ROOTS[condition]),
        "minutes_requested": minutes,
        "minutes_used": used_minutes,
        "stride": stride,
        "fps": cond.fps,
        "effective_fps": cond.fps / stride,
        "split": "todo o segundo tempo é teste",
        "link_mode": link_mode,
        "class_names": {str(k): v for k, v in config.CLASS_NAMES.items()},
        "images": len(items),
    }

    sampled = sum(r.sampled for r in reports)
    empty = sum(r.empty_labels for r in reports)
    print(f"Conjunto   : segundo_tempo/{name}")
    print(f"Minutos    : {len(used_minutes)} de {minutes} pedidos")
    print(f"Stride     : {stride} de {cond.fps} fps => {cond.fps / stride:g} fps efetivos")
    print(f"Imagens    : {len(items)}")
    if sampled:
        print(f"Labels     : {100 * (1 - empty / sampled):.0f}% com caixa (amostra de {sampled})")

    if dry_run:
        return data_yaml, manifest

    manifest_path = scenario / "manifest.json"
    if not force and manifest_path.is_file() and data_yaml.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if existing == manifest:
            print("Conjunto já montado e idêntico — reaproveitando (--force refaz).")
            return data_yaml, manifest

    img_dir = scenario / "images" / "test"
    lbl_dir = scenario / "labels" / "test"
    for directory in (img_dir, lbl_dir):
        directory.mkdir(parents=True, exist_ok=True)
        prepare.clear_leaf(directory)
    for cache in (scenario / "labels").glob("*.cache"):
        cache.unlink()

    linker = prepare.symlink_force if link_mode == prepare.LINK_SYMLINK else prepare.copy_force
    prepare.link_batch(items, img_dir, lbl_dir, linker, "teste")

    # O Ultralytics exige as chaves train e val no YAML; as três apontam para o
    # mesmo conjunto porque aqui nada é treinado. A avaliação usa split=test.
    names = "\n".join(f"  {k}: {v}" for k, v in sorted(config.CLASS_NAMES.items()))
    data_yaml.write_text(
        "# gerado por validate.py -- não editar à mão\n"
        "# segundo tempo: só avaliação, nada aqui é usado para treinar\n"
        f"path: {scenario.resolve().as_posix()}\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[OK] conjunto de teste montado em {scenario} ({link_mode})")
    return data_yaml, manifest


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #


@dataclass
class ModelToEval:
    run: str
    weight: str  # best, last, epoch010, ...
    path: Path
    condition: str
    train_imgsz: int
    train_val_epoch: int | None  # época correspondente na validação do treino
    train_val_map: float | None  # mAP50-95 dessa época na validação do treino


def run_condition(run_dir: Path) -> str:
    """Condição de treino de um run: run_meta.json, senão o prefixo do nome."""
    meta = run_dir / "run_meta.json"
    if meta.is_file():
        try:
            cond = json.loads(meta.read_text(encoding="utf-8")).get("condition")
            if cond in config.CONDITIONS:
                return cond
        except (OSError, ValueError):
            pass
    for cond in sorted(config.CONDITIONS, key=len, reverse=True):
        if run_dir.name.startswith(cond + "_"):
            return cond
    raise SystemExit(f"Não sei a condição do run {run_dir.name} (sem run_meta.json).")


def run_imgsz(run_dir: Path) -> int:
    """imgsz usado no treino, lido do args.yaml do Ultralytics."""
    args_yaml = run_dir / "args.yaml"
    if args_yaml.is_file():
        import yaml

        try:
            value = (yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}).get("imgsz")
            if value:
                return int(value)
        except (OSError, ValueError, yaml.YAMLError):
            pass
    return config.DEFAULT_IMGSZ


def train_val_reference(run_dir: Path, weight: str) -> tuple[int | None, float | None]:
    """Época e mAP50-95 que esse peso teve na validação durante o treino.

    best.pt é a época de maior mAP50-95 (critério do Ultralytics 8.4), last.pt a
    última, e epochNNN a própria época NNN.
    """
    path = run_dir / "results.csv"
    if not path.is_file():
        return None, None
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]
    key = "metrics/mAP50-95(B)"
    rows = [r for r in rows if r.get(key) not in (None, "")]
    if not rows:
        return None, None
    if weight == "best":
        row = max(rows, key=lambda r: float(r[key]))
    elif weight == "last":
        row = max(rows, key=lambda r: int(float(r["epoch"])))
    elif weight.startswith("epoch") and weight[5:].isdigit():
        wanted = int(weight[5:])
        match = [r for r in rows if int(float(r["epoch"])) == wanted]
        if not match:
            return wanted, None
        row = match[0]
    else:
        return None, None
    return int(float(row["epoch"])), float(row[key])


def resolve_models(runs: list[str], weights: list[str]) -> list[ModelToEval]:
    detect_dir = config.RUNS_DIR / "detect"
    models: list[ModelToEval] = []
    problems: list[str] = []
    for run in runs:
        run_dir = detect_dir / run
        if not run_dir.is_dir():
            problems.append(f"run não encontrado: {run_dir}")
            continue
        condition = run_condition(run_dir)
        imgsz = run_imgsz(run_dir)
        for weight in weights:
            path = run_dir / "weights" / f"{weight}.pt"
            if not path.is_file():
                problems.append(f"peso não encontrado: {path}")
                continue
            epoch, ref = train_val_reference(run_dir, weight)
            models.append(ModelToEval(run, weight, path, condition, imgsz, epoch, ref))
    if problems:
        raise SystemExit("Não dá para validar:\n" + "\n".join(f"  - {p}" for p in problems))
    return models


def all_runs() -> list[str]:
    detect_dir = config.RUNS_DIR / "detect"
    return sorted(p.name for p in detect_dir.iterdir() if (p / "weights").is_dir())


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #

SUMMARY_FIELDS = [
    "run",
    "peso",
    "condicao",
    "conjunto",
    "minutos_usados",
    "imagens",
    "stride",
    "imgsz",
    "precisao",
    "recall",
    "mAP50",
    "mAP50-95",
    "treino_epoca",
    "treino_val_mAP50-95",
    "delta_mAP50-95",
    "ms_por_imagem",
    "saida",
]


def evaluate(
    model_info: ModelToEval,
    data_yaml: Path,
    manifest: dict,
    args: argparse.Namespace,
) -> dict:
    from ultralytics import YOLO

    name = f"{model_info.run}__{model_info.weight}"
    imgsz = args.imgsz or model_info.train_imgsz
    print(f"\n=== {name}  ({CONDITION_LABELS[model_info.condition]}, imgsz {imgsz}) ===")

    model = YOLO(str(model_info.path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(VAL_RUNS_DIR),
        name=name,
        exist_ok=True,  # revalidar o mesmo peso sobrescreve a pasta dele
        plots=True,
        verbose=False,
    )

    box = metrics.box
    speed = getattr(metrics, "speed", {}) or {}
    ms = sum(float(v) for v in speed.values()) if speed else None
    row = {
        "run": model_info.run,
        "peso": model_info.weight,
        "condicao": model_info.condition,
        "conjunto": data_yaml.parent.name,
        "minutos_usados": len(manifest["minutes_used"]),
        "imagens": manifest["images"],
        "stride": manifest["stride"],
        "imgsz": imgsz,
        "precisao": round(float(box.mp), 5),
        "recall": round(float(box.mr), 5),
        "mAP50": round(float(box.map50), 5),
        "mAP50-95": round(float(box.map), 5),
        "treino_epoca": model_info.train_val_epoch if model_info.train_val_epoch is not None else "",
        "treino_val_mAP50-95": (
            round(model_info.train_val_map, 5) if model_info.train_val_map is not None else ""
        ),
        "delta_mAP50-95": (
            round(float(box.map) - model_info.train_val_map, 5)
            if model_info.train_val_map is not None
            else ""
        ),
        "ms_por_imagem": round(ms, 2) if ms is not None else "",
        "saida": str(metrics.save_dir),
    }

    out_dir = Path(metrics.save_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            dict(row, weights=str(model_info.path), data=str(data_yaml), dataset=manifest),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return row


def update_summary(rows: list[dict]) -> None:
    """Atualiza runs/val/segundo_tempo.csv, uma linha por (run, peso, conjunto)."""
    VAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if SUMMARY_CSV.is_file():
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as fh:
            # ignora linhas vazias ou quebradas (ex.: o CSV foi aberto e salvo no Excel, que põe ";;")
            existing = [r for r in csv.DictReader(fh) if r.get("run") and r.get("condicao")]
    key = lambda r: (r["run"], r["peso"], r["conjunto"])  # noqa: E731
    fresh = {key(r) for r in rows}
    merged = [r for r in existing if key(r) not in fresh] + rows
    merged.sort(key=lambda r: (r["condicao"], r["run"], r["peso"]))
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)


def print_table(rows: list[dict]) -> None:
    print("\nResultados no segundo tempo")
    header = f"{'run':32s} {'peso':9s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>8s} {'treino':>7s} {'delta':>7s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ref = r["treino_val_mAP50-95"]
        delta = r["delta_mAP50-95"]
        print(
            f"{r['run']:32s} {r['peso']:9s} {r['precisao']:6.3f} {r['recall']:6.3f} "
            f"{r['mAP50']:6.3f} {r['mAP50-95']:8.3f} "
            f"{(f'{ref:.3f}' if ref != '' else '-'):>7s} "
            f"{(f'{delta:+.3f}' if delta != '' else '-'):>7s}"
        )
    print("\n'treino' é o mAP50-95 desse peso na validação do primeiro tempo; 'delta' = segundo - primeiro.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida os detectores no segundo tempo (sem treinar).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Runs de runs/detect a validar. Padrão: " + " ".join(DEFAULT_RUNS) + ".",
    )
    which.add_argument(
        "--all-runs", action="store_true", help="Valida todas as runs de runs/detect."
    )
    parser.add_argument(
        "--condition",
        "--cond",
        dest="condition",
        choices=sorted(config.CONDITIONS),
        default=None,
        help="Valida só os modelos dessa condição. Padrão: as duas.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
        help="Pesos de cada run: best, last, epoch010, epoch025, epoch050.",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=DEFAULT_MINUTES,
        help="Usa min1 .. minN do segundo tempo.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=f"1 quadro a cada N. Padrão: {config.TARGET_FPS} fps efetivos "
        "(3 no ângulo aberto, 6 no broadcast).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Padrão: o imgsz usado no treino de cada run."
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_VAL_BATCH)
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--device", default=None, help="Ex.: 0, cpu. Padrão: automático.")
    parser.add_argument(
        "--conf", type=float, default=0.001, help="Limiar de confiança (padrão de validação)."
    )
    parser.add_argument("--iou", type=float, default=0.7, help="IoU do NMS (padrão do Ultralytics).")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Segue mesmo com pastas ou frames faltando, usando só os disponíveis.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Lê todos os labels na checagem (lento)."
    )
    parser.add_argument(
        "--link-mode",
        choices=(prepare.LINK_SYMLINK, prepare.LINK_COPY),
        default=prepare.LINK_SYMLINK,
    )
    parser.add_argument("--force", action="store_true", help="Remonta o conjunto de teste.")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Monta o conjunto de teste e para."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Só confere os dados e mostra o plano."
    )
    return parser


def main() -> None:
    config.safe_streams()
    args = build_parser().parse_args()
    if args.minutes < 1:
        raise SystemExit("--minutes precisa ser >= 1.")

    explicit_runs = args.runs is not None
    if args.all_runs:
        runs = all_runs()
    else:
        runs = args.runs if explicit_runs else list(DEFAULT_RUNS)

    if args.condition and not explicit_runs:
        # filtra pelo nome antes de procurar os pesos, para um run de outra
        # condição que ainda não existe (ex.: broadcast) não travar o comando
        runs = [r for r in runs if r.startswith(args.condition + "_")]
    models = resolve_models(runs, args.weights)

    if args.condition:
        other = sorted({m.run for m in models if m.condition != args.condition})
        if other:
            raise SystemExit(
                f"--cond {args.condition}, mas estas runs são de outra condição: "
                + ", ".join(other)
            )
    if not models:
        raise SystemExit("Nenhum modelo para validar com esses filtros.")

    print("Modelos a validar (sem treino):")
    for m in models:
        ref = f"val do treino: ép. {m.train_val_epoch}, mAP50-95 {m.train_val_map:.3f}" if m.train_val_map is not None else "sem referência do treino"
        print(f"  - {m.run}/{m.weight}.pt  [{CONDITION_LABELS[m.condition]}]  ({ref})")

    # Um conjunto de teste por condição, compartilhado pelos modelos dela.
    datasets: dict[str, tuple[Path, dict]] = {}
    for condition in sorted({m.condition for m in models}):
        stride = config.resolve_stride(condition, args.stride)
        datasets[condition] = build_test_set(
            condition,
            args.minutes,
            stride,
            strict=args.strict,
            allow_partial=args.allow_partial,
            link_mode=args.link_mode,
            force=args.force,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("\n--dry-run: nada foi montado nem avaliado.")
        return
    if args.prepare_only:
        print("\n--prepare-only: conjuntos de teste prontos, nada foi avaliado.")
        return

    rows = []
    for m in models:
        data_yaml, manifest = datasets[m.condition]
        rows.append(evaluate(m, data_yaml, manifest, args))
        update_summary(rows[-1:])  # grava a cada modelo, para não perder nada se cair

    print_table(rows)
    print(f"\n[OK] tabela em {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
