#!/usr/bin/env python3
"""Treina um detector YOLO para um cenário de N minutos do TCC.

Uso:
    python train.py 5                    # ângulo aberto, 5 min, stride 3, 50 épocas
    python train.py 40 --stride 1        # sem subamostragem
    python train.py 10 --aug scale       # segunda configuração de aumento
    python train.py 5 --prepare-only     # só monta o dataset
    python train.py 5 --no-prepare       # reusa o dataset já montado
    python train.py 5 --resume           # retoma um treino interrompido

O dataset é montado (ou reaproveitado) por prepare.py antes do treino. Os pesos
das épocas de checkpoint (10, 25 e 50 por padrão) ficam em
``runs/detect/<run>/weights/epoch0NN.pt``, além de best.pt e last.pt.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import config
import prepare


def parse_checkpoints(raw: str) -> tuple[int, ...]:
    try:
        values = sorted({int(part) for part in raw.split(",") if part.strip()})
    except ValueError:
        raise SystemExit(f"--checkpoints inválido: {raw!r} (esperado algo como 10,25,50)") from None
    if any(v < 1 for v in values):
        raise SystemExit("--checkpoints só aceita épocas >= 1.")
    return tuple(values)


def resolve_checkpoints(raw: str | None, resuming: bool, run_dir: Path) -> tuple[int, ...]:
    """Épocas de checkpoint: o que veio na CLI, senão o run anterior, senão o padrão.

    Ao retomar, o Ultralytics restaura os args dele a partir do checkpoint, mas
    os callbacks daqui usam esta lista. Sem herdar do run_meta.json, retomar sem
    repetir --checkpoints perderia em silêncio os snapshots que faltam.
    """
    if raw is not None:
        return parse_checkpoints(raw)
    if resuming:
        meta_path = run_dir / "run_meta.json"
        if meta_path.is_file():
            try:
                saved = json.loads(meta_path.read_text(encoding="utf-8")).get("checkpoint_epochs")
            except (OSError, ValueError):
                saved = None
            if saved:
                print(f"--resume: herdando checkpoints do run anterior: {saved}")
                return tuple(sorted(int(e) for e in saved))
    return tuple(config.CHECKPOINT_EPOCHS)


def check_resumable(last: Path) -> tuple[int, int]:
    """Confere se dá para retomar e devolve (próxima época, épocas alvo).

    Sem isso, um run já concluído só falha depois de o Ultralytics montar o
    treino inteiro, e com um AssertionError de quinze linhas: o strip_optimizer
    grava ``epoch: -1`` no last.pt ao fim do treino.
    """
    import torch

    try:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SystemExit(f"Não consegui ler o checkpoint {last}:\n  {exc}") from exc

    epoch = ckpt.get("epoch", -1)
    target = int((ckpt.get("train_args") or {}).get("epochs") or 0)
    if epoch is None or epoch < 0:
        raise SystemExit(
            f"Esse run já concluiu as {target} épocas — não há o que retomar:\n"
            f"  {last}\n\n"
            "Para refazer o treino, apague a pasta do run. Para continuar a partir "
            "desses pesos, use --model com esse .pt (vira um treino novo)."
        )
    return epoch + 2, target


def make_snapshot_callback(epochs_wanted: tuple[int, ...]):
    """Copia last.pt nas épocas de checkpoint.

    Usa o hook ``on_model_save``, que dispara logo depois de o Ultralytics
    gravar o last.pt daquela época. O ``on_fit_epoch_end`` seria errado aqui:
    ele dispara de novo dentro do final_eval(), ainda com trainer.epoch == 49 e
    já com o last.pt passado pelo strip_optimizer, sobrescrevendo o epoch050.

    O save_period embutido também não serve: ele nomeia com a época 0-indexada
    e sem zero-padding, então nunca produz 10/25/50.
    """

    def snapshot(trainer) -> None:
        epoch = trainer.epoch + 1  # trainer.epoch é 0-indexado
        if epoch in epochs_wanted and trainer.last.exists():
            destination = trainer.wdir / f"epoch{epoch:03d}.pt"
            shutil.copy2(trainer.last, destination)
            print(f"[checkpoint] época {epoch} -> {destination.name}")

    return snapshot


def make_strip_callback(epochs_wanted: tuple[int, ...]):
    """Remove o estado do otimizador dos snapshots, como o Ultralytics faz no best.pt."""

    def strip(trainer) -> None:
        from ultralytics.utils.torch_utils import strip_optimizer

        for epoch in epochs_wanted:
            path = trainer.wdir / f"epoch{epoch:03d}.pt"
            if path.exists():
                try:
                    strip_optimizer(path)
                except Exception as exc:  # não vale perder o treino por causa disso
                    print(f"[aviso] não consegui limpar {path.name}: {exc}")

    return strip


def make_meta_callback(meta: dict):
    """Grava run_meta.json assim que o save_dir do run existe.

    Os valores de epochs/imgsz/batch saem do trainer, não da linha de comando:
    ao retomar, o Ultralytics restaura os args do checkpoint e ignora o que veio
    da CLI, então gravar os argumentos da CLI mentiria sobre o run.
    """

    def write_meta(trainer) -> None:
        payload = dict(
            meta,
            save_dir=str(trainer.save_dir),
            epochs=trainer.args.epochs,
            imgsz=trainer.args.imgsz,
            batch=trainer.args.batch,
            # resume_training() roda dentro de _setup_train, antes deste hook
            start_epoch=trainer.start_epoch + 1,
        )
        path = Path(trainer.save_dir) / "run_meta.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return write_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina um detector YOLO para um cenário de N minutos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "minutes",
        type=int,
        help="Volume de dados em minutos (usa min1 .. minN, cumulativo).",
    )
    prepare.add_dataset_args(parser)
    parser.add_argument(
        "--aug",
        choices=sorted(config.AUG_PROFILES),
        default="default",
        help="Configuração de aumento de dados (seção 4.5 do TCC).",
    )
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="Pesos base.")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMGSZ)
    parser.add_argument(
        "--batch",
        type=int,
        default=config.DEFAULT_BATCH,
        help="Batch fixo. -1 liga o autobatch, que varia com a VRAM livre e "
        "atrapalha a comparação entre os treinos.",
    )
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--device", default=None, help="Ex.: 0, cpu. Padrão: automático.")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Épocas em que salvar um snapshot dos pesos (padrão: "
        + ",".join(str(e) for e in config.CHECKPOINT_EPOCHS)
        + "; ao retomar, herda do run_meta.json do run anterior).",
    )
    parser.add_argument("--no-prepare", action="store_true", help="Não monta o dataset antes.")
    parser.add_argument("--prepare-only", action="store_true", help="Monta o dataset e para.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma o treino a partir do last.pt do run de mesmo nome.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria feito, sem montar dataset nem treinar.",
    )
    return parser


def main() -> None:
    config.safe_streams()
    args = build_parser().parse_args()

    run_name = config.run_name(args.condition, args.minutes, args.stride, args.aug)
    project_dir = config.RUNS_DIR / "detect"
    run_dir = project_dir / run_name
    checkpoints = resolve_checkpoints(args.checkpoints, args.resume, run_dir)
    data_yaml = config.scenario_dir(args.condition, args.minutes, args.stride) / "data.yaml"

    if args.no_prepare:
        if not data_yaml.is_file():
            raise SystemExit(
                f"--no-prepare mas o dataset não existe: {data_yaml}\n"
                f"Rode sem --no-prepare, ou: python prepare.py {args.minutes} "
                f"--condition {args.condition} --stride {args.stride}"
            )
        print(f"--no-prepare: usando {data_yaml}")
    else:
        data_yaml = prepare.prepare(
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

    if args.prepare_only:
        print("\n--prepare-only: dataset pronto, treino não iniciado.")
        return

    aug = config.AUG_PROFILES[args.aug]
    weights = args.model
    epochs = args.epochs
    # resume precisa ser o CAMINHO do last.pt, não True: com um booleano o
    # Ultralytics cai em get_latest_run(), que faz glob de ./**/last*.pt e pega
    # o mais recente por data — retomaria o run errado quando houver vários.
    resume: bool | str = False
    resumed_from = 0
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            raise SystemExit(f"--resume mas não há um treino para retomar: {last}")
        resumed_from, epochs = check_resumable(last)
        weights = str(last)
        resume = str(last)

    skipped = [e for e in checkpoints if e > epochs]
    if skipped:
        print(f"\n[aviso] checkpoints além de {epochs} épocas nunca vão disparar: {skipped}")

    print()
    print(f"Run        : {run_name}")
    print(f"Modelo     : {weights}")
    if resumed_from:
        # ao retomar, o total de épocas vem do checkpoint, não da CLI
        print(f"Épocas     : {resumed_from} a {epochs} | checkpoints em {list(checkpoints)}")
    else:
        print(f"Épocas     : {epochs} | checkpoints em {list(checkpoints)}")
    print(f"imgsz/batch: {args.imgsz} / {args.batch} | workers {args.workers}")
    print(f"Aumento    : {args.aug}" + (f" {aug}" if aug else " (padrão do Ultralytics)"))
    print(f"Saída      : {run_dir}")

    if args.dry_run:
        print("\n--dry-run: treino não iniciado.")
        return

    if run_dir.exists() and not args.resume:
        raise SystemExit(
            f"\nJá existe um run com esse nome: {run_dir}\n"
            "Apague a pasta, ou use --resume para retomá-lo."
        )

    meta = {
        "run": run_name,
        "condition": args.condition,
        "minutes": args.minutes,
        "stride": args.stride,
        "aug_profile": args.aug,
        "aug_overrides": aug,
        "model": args.model,
        "checkpoint_epochs": list(checkpoints),
        "data": str(data_yaml),
        # epochs/imgsz/batch/start_epoch são preenchidos pelo callback a partir
        # do trainer, que é quem conhece os valores efetivos ao retomar
    }

    from ultralytics import YOLO

    model = YOLO(weights)
    model.add_callback("on_train_start", make_meta_callback(meta))
    model.add_callback("on_model_save", make_snapshot_callback(checkpoints))
    model.add_callback("on_train_end", make_strip_callback(checkpoints))

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        exist_ok=args.resume,
        resume=resume,
        cache=False,  # 4 GB de VRAM e milhares de PNGs; cache anularia os symlinks
        **aug,
    )

    print(f"\n[OK] treino concluído: {run_dir}")


if __name__ == "__main__":
    main()
