#!/usr/bin/env python3
"""Valida e monta o dataset YOLO de um cenário de N minutos, com symlinks.

Uso:
    python prepare.py 5                  # ângulo aberto, 5 minutos, stride 3
    python prepare.py 40 --stride 1      # sem subamostragem
    python prepare.py 10 --dry-run       # só valida e mostra as contagens

O cenário de N minutos usa as pastas ``min1 .. minN`` (cumulativo, como a seção
4.4 do TCC exige). Dentro de cada minuto o split é por bloco temporal: os
primeiros 1260 frames vão para treino e os últimos 540 para validação, o que
mantém o 70/30 sem colocar quadros quase idênticos dos dois lados e faz o split
de um minuto ser o mesmo em todos os cenários.

Nada é copiado do disco de origem: o dataset é montado com symlinks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import config

# Quantos labels ler por pasta na checagem de conteúdo. Abrir os 1800 arquivos
# de cada minuto custa minutos no disco externo; a amostra pega desvio grosseiro
# (classe errada, pasta cheia de vazios) sem esse custo. --strict lê todos.
SAMPLE_LABELS = 60

LINK_SYMLINK = "symlink"
LINK_COPY = "copy"


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #


@dataclass
class MinuteReport:
    """O que foi encontrado numa pasta min<N>."""

    minute: int
    obj_dir: Path
    expected: list[str] = field(default_factory=list)
    usable: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    zip_present: bool = False
    sampled: int = 0
    empty_labels: int = 0
    class_ids: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _expected_stems(condition: str, minute: int, minute_path: Path) -> list[str]:
    """Lista de frames que a pasta deveria ter.

    Prefere o ``train.txt`` gerado pelo próprio CVAT, que é o manifesto
    autoritativo da exportação; se ele não existir, deriva da faixa de índices
    globais do minuto.
    """
    manifest = minute_path / config.CVAT_MANIFEST
    if manifest.is_file():
        stems = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip().replace("\\", "/")
            if line.lower().endswith(config.IMG_EXT):
                stems.append(Path(line).stem)
        if stems:
            return sorted(set(stems))
    return [config.frame_stem(i) for i in config.frame_index_range(condition, minute)]


def _read_label(path: Path) -> tuple[bool, set[str]]:
    """Devolve (está vazio, ids de classe encontrados)."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return False, set()
    rows = [row for row in content.split("\n") if row.strip()]
    return (not rows), {row.split()[0] for row in rows}


def scan_minute(condition: str, minute: int, strict: bool = False) -> MinuteReport:
    """Inspeciona uma pasta min<N> sem abrir os arquivos de imagem."""
    minute_path = config.minute_dir(condition, minute)
    obj_dir = minute_path / config.OBJ_SUBDIR
    report = MinuteReport(
        minute=minute,
        obj_dir=obj_dir,
        zip_present=config.minute_zip(condition, minute).is_file(),
    )

    if not minute_path.is_dir():
        report.errors.append(f"pasta não encontrada: {minute_path}")
        return report
    if not obj_dir.is_dir():
        report.errors.append(f"sem {config.OBJ_SUBDIR}/: {obj_dir}")
        return report

    # Um único scandir por pasta: nada de abrir 1800 arquivos aqui.
    images: set[str] = set()
    labels: set[str] = set()
    with os.scandir(obj_dir) as entries:
        for entry in entries:
            name = entry.name
            if name.endswith(config.IMG_EXT):
                images.add(name[: -len(config.IMG_EXT)])
            elif name.endswith(".txt"):
                labels.add(name[:-4])

    report.expected = _expected_stems(condition, minute, minute_path)
    expected_set = set(report.expected)
    report.missing_images = sorted(expected_set - images)
    report.missing_labels = sorted(expected_set - labels)
    report.usable = sorted(expected_set & images & labels)

    if report.missing_images or report.missing_labels:
        detail = (
            f"{len(report.missing_images)} imagem(ns) e "
            f"{len(report.missing_labels)} label(s) faltando "
            f"de {len(report.expected)} esperados"
        )
        if report.zip_present:
            detail += f" — min{minute}.zip ainda está lá, extração em andamento?"
        report.errors.append(detail)

    # Checagem de conteúdo por amostragem (ou completa com --strict).
    stems = report.usable
    if stems:
        step = 1 if strict else max(1, len(stems) // SAMPLE_LABELS)
        for stem in stems[::step]:
            is_empty, ids = _read_label(obj_dir / f"{stem}.txt")
            report.sampled += 1
            report.empty_labels += int(is_empty)
            report.class_ids |= ids
        unexpected = report.class_ids - {str(i) for i in config.CLASS_NAMES}
        if unexpected:
            report.errors.append(
                f"ids de classe inesperados nos labels: {sorted(unexpected)} "
                f"(esperado apenas {sorted(config.CLASS_NAMES)})"
            )

    return report


def validate(
    condition: str,
    minutes: int,
    *,
    strict: bool = False,
    allow_partial: bool = False,
) -> list[MinuteReport]:
    """Valida min1..minN e aborta com o resumo completo se algo estiver faltando."""
    if minutes < 1:
        raise SystemExit("O número de minutos precisa ser >= 1.")

    root = config.condition_root(condition)
    if not root.is_dir():
        raise SystemExit(
            f"Raiz dos dados da condição {condition!r} não existe: {root}\n"
            "Ajuste config.CONDITIONS ou a variável de ambiente correspondente."
        )

    print(f"Validando {minutes} minuto(s) em {root} ...")
    reports = [scan_minute(condition, m, strict=strict) for m in range(1, minutes + 1)]

    broken = [r for r in reports if not r.ok]
    if broken:
        lines = [
            f"Faltam dados para o cenário de {minutes} minuto(s) "
            f"na condição {condition!r}:"
        ]
        for r in broken:
            for err in r.errors:
                lines.append(f"  - min{r.minute}: {err}")
        if allow_partial:
            lines.append("")
            lines.append("--allow-partial: seguindo só com os frames disponíveis.")
            print("\n".join(lines))
        else:
            lines.append("")
            lines.append(
                "Extraia/anote as pastas que faltam, ou use --allow-partial "
                "para montar só com o que existe."
            )
            raise SystemExit("\n".join(lines))

    return reports


# --------------------------------------------------------------------------- #
# Seleção e split
# --------------------------------------------------------------------------- #


def split_stems(
    stems: list[str], stride: int, cond: config.Condition
) -> tuple[list[str], list[str]]:
    """Separa os frames de um minuto em treino/val por bloco temporal.

    A posição dentro do minuto vem do índice global do frame; o stride é
    aplicado dentro de cada bloco para o 70/30 continuar exato.
    """
    train: list[str] = []
    val: list[str] = []
    for stem in stems:
        index = int(stem.rsplit("_", 1)[1])
        pos = index % cond.frames_per_min
        if pos < cond.train_frames:
            if pos % stride == 0:
                train.append(stem)
        elif (pos - cond.train_frames) % stride == 0:
            val.append(stem)
    return train, val


def collect_items(
    reports: list[MinuteReport], stride: int, cond: config.Condition
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Monta as listas (pasta de origem, stem) de treino e validação."""
    train: list[tuple[Path, str]] = []
    val: list[tuple[Path, str]] = []
    for report in reports:
        t, v = split_stems(report.usable, stride, cond)
        train.extend((report.obj_dir, stem) for stem in t)
        val.extend((report.obj_dir, stem) for stem in v)
    return train, val


# --------------------------------------------------------------------------- #
# Montagem
# --------------------------------------------------------------------------- #


def symlink_force(src: Path, dst: Path) -> None:
    if os.path.lexists(dst):
        os.unlink(dst)
    try:
        os.symlink(src, dst)
    except OSError as exc:  # WinError 1314: privilégio não mantido
        raise SystemExit(
            f"Não consegui criar symlink:\n  {dst} -> {src}\n  {exc}\n\n"
            "No Windows, criar symlink exige o Modo de Desenvolvedor ligado "
            "(Configurações > Privacidade e segurança > Para desenvolvedores) "
            "ou um terminal como administrador.\n"
            "Alternativa: rode com --link-mode copy (copia os arquivos, "
            "ocupando espaço em disco)."
        ) from exc


def copy_force(src: Path, dst: Path) -> None:
    if os.path.lexists(dst):
        os.unlink(dst)
    shutil.copy2(src, dst)


def clear_leaf(path: Path) -> None:
    """Esvazia uma pasta folha (links e arquivos), preservando a pasta."""
    if not path.is_dir():
        return
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_symlink() or entry.is_file(follow_symlinks=False):
                os.unlink(entry.path)


def link_batch(
    items: list[tuple[Path, str]],
    img_dir: Path,
    lbl_dir: Path,
    linker,
    label: str,
) -> None:
    total = len(items)
    for i, (obj_dir, stem) in enumerate(items, start=1):
        linker(obj_dir / f"{stem}{config.IMG_EXT}", img_dir / f"{stem}{config.IMG_EXT}")
        linker(obj_dir / f"{stem}.txt", lbl_dir / f"{stem}.txt")
        if i % 5000 == 0 or i == total:
            print(f"  {label}: {i}/{total}")


def write_data_yaml(scenario: Path) -> Path:
    """Escreve o data.yaml do cenário com caminho absoluto e com letra de unidade.

    Caminho relativo aqui seria resolvido contra o DATASETS_DIR das settings do
    Ultralytics, que aponta para outro projeto nesta máquina.
    """
    names = "\n".join(f"  {k}: {v}" for k, v in sorted(config.CLASS_NAMES.items()))
    path = scenario / "data.yaml"
    path.write_text(
        "# gerado por prepare.py -- não editar à mão\n"
        f"path: {scenario.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return path


def _manifest(
    condition: str, minutes: int, stride: int, link_mode: str, n_train: int, n_val: int
) -> dict:
    cond = config.get_condition(condition)
    return {
        "condition": condition,
        "data_root": str(config.condition_root(condition)),
        "minutes": minutes,
        "minute_folders": [f"min{i}" for i in range(1, minutes + 1)],
        "stride": stride,
        "split": {
            "rule": "bloco temporal por minuto",
            "fps": cond.fps,
            "frames_per_min": cond.frames_per_min,
            "train_frames": cond.train_frames,
            "val_frames": cond.val_frames,
            "effective_fps": cond.fps / stride,
        },
        "link_mode": link_mode,
        "class_names": {str(k): v for k, v in config.CLASS_NAMES.items()},
        "counts": {"train": n_train, "val": n_val},
    }


def prepare(
    condition: str = config.DEFAULT_CONDITION,
    minutes: int = 5,
    stride: int | None = None,
    *,
    clean: bool = False,
    force: bool = False,
    strict: bool = False,
    allow_partial: bool = False,
    link_mode: str = LINK_SYMLINK,
    dry_run: bool = False,
    epochs: int = config.DEFAULT_EPOCHS,
) -> Path:
    """Valida, monta e devolve o caminho do data.yaml do cenário."""
    stride = config.resolve_stride(condition, stride)
    cond = config.get_condition(condition)

    reports = validate(condition, minutes, strict=strict, allow_partial=allow_partial)
    train_items, val_items = collect_items(reports, stride, cond)

    if not train_items:
        raise SystemExit("Nenhum frame de treino selecionado — verifique o --stride.")

    scenario = config.scenario_dir(condition, minutes, stride)
    data_yaml_path = scenario / "data.yaml"
    manifest_data = _manifest(
        condition, minutes, stride, link_mode, len(train_items), len(val_items)
    )

    sampled = sum(r.sampled for r in reports)
    empty = sum(r.empty_labels for r in reports)
    print()
    print(f"Cenário    : {config.scenario_name(condition, minutes, stride)}")
    print(f"Minutos    : min1 .. min{minutes} ({len(reports)} pasta(s))")
    print(
        f"Stride     : {stride} de {cond.fps} fps "
        f"=> {cond.fps / stride:g} fps efetivos"
    )
    print(f"Treino/val : {len(train_items)} / {len(val_items)} imagens")
    if sampled:
        pct = 100 * (1 - empty / sampled)
        print(f"Labels     : {pct:.0f}% com caixa (amostra de {sampled})")
    fast, slow = config.estimate_hours(len(train_items), epochs)
    print(
        f"Estimativa : ~{fast:.1f}-{slow:.1f} h para {epochs} épocas "
        f"(limitado pela GPU / pela leitura do disco)"
    )
    print(f"Destino    : {scenario}")

    if dry_run:
        print("\n--dry-run: nada foi escrito.")
        return data_yaml_path

    manifest_path = scenario / "manifest.json"
    if not clean and not force and manifest_path.is_file() and data_yaml_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if existing == manifest_data:
            print("\nDataset já montado e idêntico ao pedido — pulando (--force refaz).")
            return data_yaml_path

    if clean and scenario.is_dir():
        print(f"\n--clean: removendo {scenario}")
        shutil.rmtree(scenario)

    images_train = scenario / "images" / "train"
    images_val = scenario / "images" / "val"
    labels_train = scenario / "labels" / "train"
    labels_val = scenario / "labels" / "val"
    for directory in (images_train, images_val, labels_train, labels_val):
        directory.mkdir(parents=True, exist_ok=True)
        clear_leaf(directory)

    # O cache de labels do Ultralytics fica em labels/*.cache e descreve o
    # conjunto anterior de arquivos; remover evita um rescan confuso.
    for cache in (scenario / "labels").glob("*.cache"):
        cache.unlink()

    linker = symlink_force if link_mode == LINK_SYMLINK else copy_force
    print()
    link_batch(train_items, images_train, labels_train, linker, "treino")
    link_batch(val_items, images_val, labels_val, linker, "val   ")

    write_data_yaml(scenario)
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n[OK] dataset montado em {scenario} ({link_mode})")
    return data_yaml_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Flags compartilhadas entre prepare.py e train.py."""
    parser.add_argument(
        "--condition",
        choices=sorted(config.CONDITIONS),
        default=config.DEFAULT_CONDITION,
        help="Condição de captura.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Pega 1 quadro a cada N. Padrão: o que deixa a amostragem em "
        "{} fps na condição escolhida (ang=3, broadcast=6).".format(config.TARGET_FPS),
    )
    parser.add_argument("--clean", action="store_true", help="Apaga a pasta do cenário antes.")
    parser.add_argument("--force", action="store_true", help="Remonta mesmo se o manifest bater.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Lê todos os labels na validação, não só uma amostra (lento).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Segue mesmo com frames faltando, usando só os disponíveis.",
    )
    parser.add_argument(
        "--link-mode",
        choices=(LINK_SYMLINK, LINK_COPY),
        default=LINK_SYMLINK,
        help="Como popular o dataset. copy só se symlink não for possível.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monta o dataset YOLO de um cenário de N minutos com symlinks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "minutes",
        type=int,
        help="Volume de dados em minutos (usa min1 .. minN, cumulativo).",
    )
    add_dataset_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Só valida e mostra as contagens.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help="Usado apenas na estimativa de tempo.",
    )
    return parser


def main() -> None:
    config.safe_streams()
    args = build_parser().parse_args()
    prepare(
        condition=args.condition,
        minutes=args.minutes,
        stride=args.stride,
        clean=args.clean,
        force=args.force,
        strict=args.strict,
        allow_partial=args.allow_partial,
        link_mode=args.link_mode,
        dry_run=args.dry_run,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
