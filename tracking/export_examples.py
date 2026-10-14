"""
Tira algumas imagens de exemplo do vídeo de uma rodada (o video.mp4 gravado com --save-video),
uma para cada tipo de frame da avaliação: acerto, perda, troca e alarme falso.

Em cada imagem, além do que o run.py já desenhou (cinza, amarelo, verde), desenha a caixa do
gabarito em magenta e escreve embaixo o rótulo do frame e o IoU.

Salva em results/<captura>/mot_custom/<algoritmo>_<config>/examples/ (apaga as imagens antigas dessa pasta antes).

Rodar (na raiz do projeto):
    python tracking/export_examples.py --capture broadcast
        sem --algorithm/--config: usa a rodada com maior acerto dessa captura (lido dos evaluation.json)
    python tracking/export_examples.py --capture ang --algorithm deepsort --config default
        escolhe a rodada na mão
"""
import argparse
import json
from pathlib import Path

import cv2

from build_ground_truth import CAPTURES as GROUND_TRUTH
from evaluate import read_mot

ROOT = Path(__file__).resolve().parent.parent  # raiz do projeto

LABELS = ["acerto", "perda", "troca", "deriva", "alarme_falso"]
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)


def pick_example_frames(rows):
    """
    Para cada rótulo, acha os blocos de frames seguidos com esse rótulo e escolhe o meio de um
    bloco longo (o 3º mais longo, para não pegar sempre o caso mais extremo).
    rows = lista de (frame, rótulo, iou). Devolve {rótulo: (frame, iou)}.
    """
    blocks = {label: [] for label in LABELS}
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i][1] != rows[start][1]:
            label = rows[start][1]
            if label in blocks:
                blocks[label].append(rows[start:i])
            start = i

    chosen = {}
    for label, label_blocks in blocks.items():
        if not label_blocks:
            continue
        label_blocks.sort(key=len, reverse=True)
        block = label_blocks[min(2, len(label_blocks) - 1)]
        frame, _, overlap = block[len(block) // 2]
        chosen[label] = (frame, overlap)
    return chosen


def find_best_run(capture):
    """A rodada da captura com maior acerto (só as que têm video.mp4). Devolve a pasta dela."""
    best_dir, best_accuracy = None, -1
    for run_dir in sorted((ROOT / "results" / capture / "mot_custom").iterdir()):
        evaluation_file = run_dir / "evaluation.json"
        if not evaluation_file.exists() or not (run_dir / "video.mp4").exists():
            continue
        accuracy = json.loads(evaluation_file.read_text(encoding="utf-8"))["accuracy"]
        print(f"  {run_dir.name:20s} acerto {100 * accuracy:.1f}%")
        if accuracy > best_accuracy:
            best_dir, best_accuracy = run_dir, accuracy
    return best_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["ang", "broadcast"], required=True)
    parser.add_argument("--algorithm", choices=["bytetrack", "deepsort", "sort"], help="padrão: a rodada com maior acerto")
    parser.add_argument("--config", choices=["default", "custom"], help="padrão: a rodada com maior acerto")
    args = parser.parse_args()

    if args.algorithm and args.config:
        run_dir = ROOT / "results" / args.capture / "mot_custom" / f"{args.algorithm}_{args.config}"
    else:
        print(f"Procurando a rodada com maior acerto em {args.capture}:")
        run_dir = find_best_run(args.capture)
        if run_dir is None:
            print("Nenhuma rodada com evaluation.json e video.mp4.")
            return
        print(f"Melhor: {run_dir.name}")
        args.algorithm, args.config = run_dir.name.split("_")
    video_path = run_dir / "video.mp4"
    if not video_path.exists():
        print(f"Não achei {video_path}. Essa rodada foi feita com --save-video?")
        return

    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    ground_truth = read_mot(GROUND_TRUTH[args.capture]["output"] / "gt.txt")

    rows = []
    for line in (run_dir / "evaluation_frames.csv").read_text(encoding="utf-8").splitlines()[1:]:
        frame, _, _, overlap, label = line.split(",")
        rows.append((int(frame), label, float(overlap)))
    examples = pick_example_frames(rows)

    out_dir = run_dir / "examples"
    out_dir.mkdir(exist_ok=True)
    for old_image in out_dir.glob("*.jpg"):  # imagens de uma exportação anterior
        old_image.unlink()
    video = cv2.VideoCapture(str(video_path))

    for label, (frame, overlap) in examples.items():
        # frame do MOT começa em 1; o vídeo da rodada começa no start_frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame - 1 - run_info["start_frame"])
        ok, image = video.read()
        if not ok:
            print(f"Não consegui ler o frame {frame} do vídeo.")
            continue

        # gabarito em magenta
        if frame in ground_truth:
            x1, y1, x2, y2 = (int(v) for v in ground_truth[frame][0][1])
            cv2.rectangle(image, (x1, y1), (x2, y2), MAGENTA, 2)
            cv2.putText(image, "GROUND TRUTH", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1)

        # faixa embaixo com o rótulo
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, height - 50), (width, height), (0, 0, 0), -1)
        cv2.putText(image, f"{label.upper()}   |   frame {frame}   |   IoU with ground truth {overlap:.2f}   |   "
                           f"{args.capture} {args.algorithm} {args.config}",
                    (15, height - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

        # reduz para 1280 de largura (imagem mais leve)
        scale = 1280 / width
        image = cv2.resize(image, (1280, int(height * scale)))
        out_file = out_dir / f"{label}_frame{frame}.jpg"
        cv2.imwrite(str(out_file), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{label}: frame {frame} (IoU {overlap:.2f}) -> {out_file}")

    video.release()


if __name__ == "__main__":
    main()
