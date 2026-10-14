"""
Avalia a trajetória do jogador alvo de uma rodada contra o gabarito do CVAT, frame a frame.
O run.py chama isso sozinho no final de cada rodada.

Usa, da pasta da rodada:
  trajectory.txt   a saída do sistema (a caixa do jogador alvo em cada frame)
  tracks.txt       todas as pessoas rastreadas (para separar troca de deriva)
  run_info.json    quais frames foram processados e o fps

Cada frame recebe um rótulo:
  acerto        jogador em campo e a caixa do sistema bate com a do gabarito (IoU >= 0.5)
  perda         jogador em campo e o sistema não mostrou caixa
  troca         jogador em campo, a caixa está em outro lugar, em cima de outra pessoa
  deriva        jogador em campo, a caixa está em outro lugar, sem ninguém embaixo
  ausente       jogador fora de cena e o sistema também não mostrou caixa (acerto)
  alarme_falso  jogador fora de cena e o sistema mostrou caixa

Depois conta:
  acerto (%)    frames de acerto / frames com o jogador em campo
  perdas/trocas cada bloco seguido de frames conta como 1 vez
  recuperação   uma falha começa no primeiro frame errado e termina no próximo acerto.
                Frames com o jogador fora de cena não contam no tempo.

Salva na pasta da rodada:
  evaluation.txt          o resumo, para ler
  evaluation.json         os mesmos números, para juntar as rodadas numa tabela depois
  evaluation_frames.csv   o rótulo de cada frame (para achar onde ele errou)

Para avaliar de novo uma rodada que já existe (na raiz do projeto):
    python tracking/evaluate.py --capture ang --algorithm bytetrack --config default

Para avaliar uma pasta qualquer, fora de mot_custom/ (ex.: as rodadas de SOT):
    python tracking/evaluate.py --capture broadcast --run-dir results/broadcast/sot/csrt_once
    python tracking/evaluate.py --capture broadcast --run-dir results/broadcast/sot/*

Rodadas sem tracks.txt (as de SOT) usam as detecções de pessoas de
results/<captura>/detections/people.txt para separar troca de deriva.
"""
import argparse
import json
import statistics
from pathlib import Path

from build_ground_truth import CAPTURES as GROUND_TRUTH, build_ground_truth

ROOT = Path(__file__).resolve().parent.parent  # raiz do projeto

CORRECT_IOU = 0.5   # IoU mínimo com o gabarito para contar como acerto
PERSON_IOU = 0.5    # IoU mínimo com outra pessoa para a caixa errada contar como troca (senão é deriva)


def iou(a, b):
    """Quanto duas caixas [x1, y1, x2, y2] se sobrepõem: 0 = nada, 1 = idênticas."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def read_mot(path):
    """Lê um arquivo MOT. Devolve {frame: [(id, [x1, y1, x2, y2]), ...]}."""
    boxes = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        values = line.strip().split(",")
        if len(values) < 6:
            continue
        frame, track_id = int(float(values[0])), int(float(values[1]))
        x, y, w, h = (float(v) for v in values[2:6])
        boxes.setdefault(frame, []).append((track_id, [x, y, x + w, y + h]))
    return boxes


def read_people_detections(path, frames_wanted):
    """Lê as detecções de pessoas do YOLO pronto (results/<captura>/detections/people.txt).

    Formato: frame,x1,y1,x2,y2,score — com o frame começando em 0, não em 1 como o MOT.
    Só guarda os frames pedidos: o arquivo do broadcast tem cerca de um milhão de linhas.
    Devolve a mesma forma do read_mot, com id -1 (são detecções, não rastros).
    """
    boxes = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            values = line.split(",")
            if len(values) < 5:
                continue
            frame = int(float(values[0])) + 1  # base 0 -> base 1 (MOT)
            if frame not in frames_wanted:
                continue
            boxes.setdefault(frame, []).append((-1, [float(v) for v in values[1:5]]))
    return boxes


def count_blocks(labels, name):
    """Quantos blocos seguidos de 'name'. Ex.: perda perda acerto perda = 2 vezes."""
    blocks = 0
    previous = None
    for label in labels:
        if label == name and previous != name:
            blocks += 1
        previous = label
    return blocks


def evaluate(capture, run_dir):
    run_dir = Path(run_dir)
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    fps = run_info["fps"]
    # No formato MOT o frame começa em 1 (frame 0 do vídeo = frame 1 do arquivo).
    # O run.py grava start_frame no índice do OpenCV (base 0) e precisa do +1; o
    # run_sot.py já grava no índice do MOT e marca isso com "frame_base": 1.
    first_frame = run_info["start_frame"] + (1 if run_info.get("frame_base", 0) == 0 else 0)
    last_frame = run_info["end_frame"]

    gt_file = GROUND_TRUTH[capture]["output"] / "gt.txt"
    if not gt_file.exists():
        print("O gabarito unificado ainda não existe. Criando agora...")
        build_ground_truth(capture)
    ground_truth = read_mot(gt_file)
    trajectory = read_mot(run_dir / "trajectory.txt")

    # "Caixa errada" se divide em troca (caiu em cima de outra pessoa) e deriva
    # (caiu no vazio), e para isso é preciso saber quem mais estava no frame.
    # As rodadas MOT têm o tracks.txt; as de SOT, não — aí valem as detecções
    # cruas do YOLO pronto, que respondem a mesma pergunta.
    frames_to_check = {f for f in range(first_frame, last_frame + 1)
                       if f in trajectory and f in ground_truth}
    tracks_file = run_dir / "tracks.txt"
    detections_file = ROOT / "results" / capture / "detections" / "people.txt"
    if tracks_file.exists():
        people = read_mot(tracks_file)
    elif detections_file.exists():
        print(f"Sem tracks.txt nesta rodada; usando as detecções de pessoas de {detections_file}")
        people = read_people_detections(detections_file, frames_to_check)
    else:
        print("AVISO: sem tracks.txt e sem detecções de pessoas — toda caixa errada vira deriva.")
        people = {}

    # 1. um rótulo para cada frame
    labels = []
    rows = ["frame,player_visible,system_id,iou,label"]
    for frame in range(first_frame, last_frame + 1):
        player_visible = frame in ground_truth
        has_box = frame in trajectory
        overlap = 0.0
        system_id = ""

        if player_visible and has_box:
            system_id, system_box = trajectory[frame][0]
            gt_box = ground_truth[frame][0][1]
            overlap = iou(system_box, gt_box)
            if overlap >= CORRECT_IOU:
                label = "acerto"
            else:
                # caixa errada: caiu em cima de outra pessoa (troca) ou do nada (deriva)?
                on_a_person = any(iou(system_box, person_box) >= PERSON_IOU for _, person_box in people.get(frame, []))
                label = "troca" if on_a_person else "deriva"
        elif player_visible:
            label = "perda"
        elif has_box:
            label = "alarme_falso"
            system_id = trajectory[frame][0][0]
        else:
            label = "ausente"

        labels.append(label)
        rows.append(f"{frame},{int(player_visible)},{system_id},{overlap:.3f},{label}")

    # 2. recuperação: cada falha vai do primeiro frame errado até o próximo acerto
    #    (frames com o jogador fora de cena não contam)
    recovered_frames = []  # duração de cada falha que se recuperou, em frames
    not_recovered = 0
    current = None         # frames da falha em andamento
    for label in labels:
        if label in ("ausente", "alarme_falso"):
            continue
        if current is None:
            if label != "acerto":
                current = 1
        elif label == "acerto":
            recovered_frames.append(current)
            current = None
        else:
            current += 1
    if current is not None:
        not_recovered = 1
    recovery_seconds = [frames / fps for frames in recovered_frames]

    # 3. resumo
    counts = {name: labels.count(name) for name in ("acerto", "perda", "troca", "deriva", "ausente", "alarme_falso")}
    visible = counts["acerto"] + counts["perda"] + counts["troca"] + counts["deriva"]
    summary = {
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frames": len(labels),
        "frames_player_visible": visible,
        "frames_by_label": counts,
        "accuracy": round(counts["acerto"] / visible, 4) if visible else None,
        "lost_times": count_blocks(labels, "perda"),
        "switch_times": count_blocks(labels, "troca"),
        "drift_times": count_blocks(labels, "deriva"),
        "false_alarm_times": count_blocks(labels, "alarme_falso"),
        "failures": len(recovered_frames) + not_recovered,
        "failures_recovered": len(recovered_frames),
        "failures_not_recovered": not_recovered,
        "recovery_seconds_mean": round(statistics.mean(recovery_seconds), 3) if recovery_seconds else None,
        "recovery_seconds_median": round(statistics.median(recovery_seconds), 3) if recovery_seconds else None,
        "recovery_seconds_max": round(max(recovery_seconds), 3) if recovery_seconds else None,
        "correct_iou": CORRECT_IOU,
        "person_iou": PERSON_IOU,
    }

    def pct(n):
        return f"{100 * n / visible:.1f}%" if visible else "-"

    text = [
        f"Avaliados os frames {first_frame} a {last_frame} ({len(labels)} frames, jogador em campo em {visible})",
        f"  Acerto:        {counts['acerto']} frames ({pct(counts['acerto'])})",
        f"  Perda:         {counts['perda']} frames ({pct(counts['perda'])}), {summary['lost_times']} vezes",
        f"  Troca:         {counts['troca']} frames ({pct(counts['troca'])}), {summary['switch_times']} vezes",
        f"  Deriva:        {counts['deriva']} frames ({pct(counts['deriva'])}), {summary['drift_times']} vezes",
        f"  Fora de cena:  {counts['ausente']} frames sem caixa (certo), "
        f"{counts['alarme_falso']} com caixa (alarme falso, {summary['false_alarm_times']} vezes)",
        f"  Falhas:        {summary['failures']}, recuperou {len(recovered_frames)}, não recuperou {not_recovered}",
    ]
    if recovery_seconds:
        text.append(f"  Recuperação:   média {summary['recovery_seconds_mean']:.2f} s, "
                    f"mediana {summary['recovery_seconds_median']:.2f} s, máximo {summary['recovery_seconds_max']:.2f} s")

    (run_dir / "evaluation_frames.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "evaluation.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n" + "\n".join(text))
    return summary


def expand_run_dirs(patterns):
    """Resolve o que veio em --run-dir numa lista de pastas de rodada.

    O cmd.exe e o PowerShell não expandem curinga (só o bash expande), então o
    glob é feito aqui. Também aceita a pasta-mãe: apontar para results/<captura>/sot
    vale por todas as rodadas dentro dela.
    """
    found = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        if not matches:
            raise SystemExit(f"Nada encontrado para: {pattern}")
        for path in matches:
            if not path.is_dir():
                raise SystemExit(f"Não é uma pasta: {path}")
            if (path / "run_info.json").exists():
                found.append(path)
                continue
            # pasta-mãe: pega as subpastas que são rodadas
            children = sorted(c for c in path.iterdir() if (c / "run_info.json").exists())
            if not children:
                raise SystemExit(f"Sem run_info.json em {path} nem nas subpastas dela.")
            found.extend(children)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["ang", "broadcast"], default="ang")
    parser.add_argument("--algorithm", choices=["bytetrack", "deepsort", "sort"], default="bytetrack")
    parser.add_argument("--config", choices=["default", "custom"], default="default")
    parser.add_argument("--run-dir", nargs="+", default=None, metavar="PASTA",
                        help="avalia uma pasta de rodada qualquer, fora de mot_custom/ "
                             "(ex.: results/broadcast/sot/csrt_once). Aceita várias.")
    args = parser.parse_args()
    if args.run_dir:
        for run_dir in expand_run_dirs(args.run_dir):
            print()
            print(f"=== {run_dir} ===")
            evaluate(args.capture, run_dir)
    else:
        evaluate(args.capture, ROOT / "results" / args.capture / "mot_custom" / f"{args.algorithm}_{args.config}")


if __name__ == "__main__":
    main()
