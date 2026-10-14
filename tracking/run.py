"""
Rastreia o jogador alvo com a proposta do TCC (YOLO pronto + rastreamento + teu YOLO),
mostra as decisões na tela, salva a trajetória e avalia contra o gabarito no final.

É o pseudocódigo, frame a frame:
  1. YOLO pronto acha todas as pessoas
  2. o algoritmo de rastreamento (SORT, DeepSORT ou ByteTrack) dá um ID para cada pessoa
  3. teu YOLO acha o jogador alvo
  4. o rastro que mais sobrepõe (IoU) a caixa do teu YOLO vira o jogador alvo (se IoU >= 0.5)
  5. a caixa do jogador alvo é a do rastro com esse ID (se ele sumiu, perdeu)

Os dois YOLOs dão sempre as mesmas caixas (mesmo vídeo, mesmo modelo), então só rodam na
primeira vez: as caixas ficam salvas em results/<captura>/detections/ e as rodadas seguintes
só leem esses arquivos. Se pedir um trecho que ainda não foi detectado, os YOLOs rodam de novo.

Na tela:
  cinza     todas as pessoas, com o ID que o rastreador deu
  amarelo   o que o teu YOLO achou como jogador alvo
  verde     a decisão final: quem o sistema diz que é o jogador alvo
  texto     o que aconteceu em cada passo neste frame

Cada rodada salva em results/<captura>/mot_custom/<algoritmo>_<config>/:
  trajectory.txt          a caixa do jogador alvo em cada frame (formato MOT, frame começa em 1)
  tracks.txt              todas as pessoas que o rastreador devolveu (formato MOT)
  run_info.json           parâmetros, frames processados, tempo e FPS
  evaluation.txt/.json    a avaliação contra o gabarito (feita no final)
  evaluation_frames.csv   o rótulo de cada frame
  video.mp4               só com --save-video

Rodar (na raiz do projeto):
    python tracking/run.py --capture ang --algorithm bytetrack --config default
    python tracking/run.py --capture broadcast --algorithm deepsort --config default --no-window
    python tracking/run.py --capture ang --algorithm sort --config default --end-frame 3600   (2 min, para testar)
    python tracking/run.py ... --save-video                                                  (grava o vídeo com os desenhos)

Teclas (com janela): espaço pausa/continua, q (ou Esc) para a rodada antes do fim.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from evaluate import evaluate
from tracker_configs import TRACKER_CONFIGS
from trackers import create_tracker

ROOT = Path(__file__).resolve().parent.parent  # raiz do projeto

VIDEOS = {
    "ang": ROOT / "angulo_aberto.mp4",
    "broadcast": ROOT / "broadcast.mp4",
}
TARGET_PLAYER_MODELS = {
    "ang": ROOT / "runs" / "detect" / "ang_40min_s3_default" / "weights" / "epoch025.pt",
    "broadcast": ROOT / "runs" / "detect" / "broadcast_40min_s6_default" / "weights" / "epoch025.pt",
}
PEOPLE_MODEL = ROOT / "yolo11n.pt"   # YOLO pronto, acha qualquer pessoa

PEOPLE_MIN_CONF = 0.1   # o YOLO de pessoas manda tudo acima disso; cada rastreador filtra com o seu limiar
TARGET_MIN_CONF = 0.5   # o teu YOLO só "vale" com pelo menos essa confiança
MIN_IOU = 0.5           # sobreposição mínima entre a caixa do teu YOLO e um rastro

GRAY = (160, 160, 160)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)


def iou(a, b):
    """Quanto duas caixas [x1, y1, x2, y2] se sobrepõem: 0 = nada, 1 = idênticas."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def mot_line(frame_number, track_id, box):
    """Uma linha no formato MOT. O frame do MOT começa em 1, o do OpenCV em 0."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return f"{frame_number + 1},{track_id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n"


def draw_box(image, box, color, thickness, label=None):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_panel(image, lines):
    """Caixa preta no canto com o texto das decisões (OpenCV não escreve acento)."""
    height = 30 + 30 * len(lines)
    cv2.rectangle(image, (0, 0), (900, height), (0, 0, 0), -1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(image, text, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def ask_to_run_again(run_dir):
    """Se a rodada já existe, pergunta se é para rodar de novo."""
    info_file = run_dir / "run_info.json"
    if not info_file.exists():
        return True
    info = json.loads(info_file.read_text(encoding="utf-8"))
    print(f"Esta rodada já existe ({info['finished_at']}, frames {info['start_frame']} a {info['end_frame'] - 1}).")
    answer = input("Rodar de novo? Isso apaga os resultados antigos. [s/N] ")
    return answer.strip().lower() in ("s", "sim")


# ------------------------------------------------------------------ detecções salvas
# results/<captura>/detections/
#   people.txt         frame, x1, y1, x2, y2, conf   (YOLO de pessoas; frame do vídeo, começando em 0)
#   target_player.txt  frame, x1, y1, x2, y2, conf   (teu YOLO)
#   info.json          trecho detectado, modelos, limiares e quanto tempo os YOLOs levaram

def detection_settings(capture):
    """O que, se mudar, obriga a rodar os YOLOs de novo."""
    return {
        "people_model": str(PEOPLE_MODEL),
        "target_player_model": str(TARGET_PLAYER_MODELS[capture]),
        "people_min_conf": PEOPLE_MIN_CONF,
        "target_min_conf": TARGET_MIN_CONF,
    }


def read_boxes(path):
    """Lê um arquivo de caixas salvo. Devolve {frame: [[x1, y1, x2, y2, conf], ...]}."""
    boxes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        frame, x1, y1, x2, y2, conf = line.split(",")
        boxes.setdefault(int(frame), []).append([float(x1), float(y1), float(x2), float(y2), float(conf)])
    return boxes


def load_saved_detections(capture, start_frame, end_frame):
    """Se as detecções salvas cobrem o trecho pedido, devolve (info, pessoas, jogador alvo). Senão, None."""
    folder = ROOT / "results" / capture / "detections"
    if not (folder / "info.json").exists():
        return None
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    if info["settings"] != detection_settings(capture):
        print("As detecções salvas são de outro modelo ou limiar. Os YOLOs vão rodar de novo.")
        return None
    if info["start_frame"] > start_frame or info["end_frame"] < end_frame:
        print(f"As detecções salvas só cobrem os frames {info['start_frame']} a {info['end_frame'] - 1}. "
              f"Os YOLOs vão rodar de novo.")
        return None
    return info, read_boxes(folder / "people.txt"), read_boxes(folder / "target_player.txt")


def save_detections(capture, start_frame, end_frame, people_lines, target_lines, people_seconds, target_seconds):
    folder = ROOT / "results" / capture / "detections"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "people.txt").write_text("".join(people_lines), encoding="utf-8")
    (folder / "target_player.txt").write_text("".join(target_lines), encoding="utf-8")
    info = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "settings": detection_settings(capture),
        "people_seconds": round(people_seconds, 1),
        "target_player_seconds": round(target_seconds, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detecções salvas em {folder} (as próximas rodadas não precisam rodar os YOLOs)")


def box_line(frame_number, box, conf):
    x1, y1, x2, y2 = (float(v) for v in box)
    return f"{frame_number},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{float(conf):.4f}\n"


# ------------------------------------------------------------------ rodada

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["ang", "broadcast"], required=True)
    parser.add_argument("--algorithm", choices=["bytetrack", "deepsort", "sort"], required=True)
    parser.add_argument("--config", choices=["default", "custom"], required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="até qual frame rodar (padrão: vídeo inteiro)")
    parser.add_argument("--no-window", action="store_true", help="não mostra a janela (mais rápido)")
    parser.add_argument("--save-video", action="store_true", help="grava o vídeo com os desenhos")
    args = parser.parse_args()

    tracker_params = TRACKER_CONFIGS[args.algorithm][args.config]
    if tracker_params is None:
        print(f"A configuração '{args.config}' do {args.algorithm} ainda não foi definida. "
              f"Preenche em tracking/tracker_configs.py.")
        return

    run_dir = ROOT / "results" / args.capture / "mot_custom" / f"{args.algorithm}_{args.config}"
    if not ask_to_run_again(run_dir):
        print("Ok, nada foi feito.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rodada: {args.capture} / {args.algorithm} / {args.config}. Salvando em {run_dir}")

    video = cv2.VideoCapture(str(VIDEOS[args.capture]))
    video.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(args.end_frame or total_frames, total_frames)

    # Detecções salvas ou YOLOs rodando agora?
    saved = load_saved_detections(args.capture, args.start_frame, end_frame)
    if saved is not None:
        detections_info, saved_people, saved_target = saved
        print("Usando as detecções salvas (os YOLOs não vão rodar).")
    else:
        from ultralytics import YOLO
        print("Rodando os YOLOs (as caixas vão ficar salvas para as próximas rodadas).")
        people_model = YOLO(str(PEOPLE_MODEL))
        target_model = YOLO(str(TARGET_PLAYER_MODELS[args.capture]))
        people_lines = []   # para salvar as caixas no final
        target_lines = []
        people_seconds = 0.0
        target_seconds = 0.0

    # Só precisa ler a imagem do vídeo se: os YOLOs vão rodar, é o DeepSORT (usa aparência),
    # ou vai mostrar/gravar o vídeo. Senão o laço anda só pelas caixas salvas, bem mais rápido.
    need_image = saved is None or args.algorithm == "deepsort" or not args.no_window or args.save_video

    tracker = create_tracker(args.algorithm, tracker_params, fps, width, height)

    trajectory_file = open(run_dir / "trajectory.txt", "w")
    tracks_file = open(run_dir / "tracks.txt", "w")
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(run_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    window = "Target player tracking"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1600, 900)

    target_id = None
    id_changes = 0
    frames_with_box = 0
    frame_number = args.start_frame
    paused = False
    start_time = time.perf_counter()

    while frame_number < end_frame:
        if not paused:
            frame = None
            if need_image:
                ok, frame = video.read()
                if not ok:
                    break

            # 1. YOLO pronto acha todas as pessoas (ou lê as caixas salvas)
            if saved is not None:
                detections = saved_people.get(frame_number, [])  # cada uma: [x1, y1, x2, y2, conf]
            else:
                t0 = time.perf_counter()
                people_result = people_model.predict(frame, classes=[0], conf=PEOPLE_MIN_CONF, verbose=False)[0]
                people_seconds += time.perf_counter() - t0
                detections = []
                for box, conf in zip(people_result.boxes.xyxy.cpu().numpy(), people_result.boxes.conf.cpu().numpy()):
                    detections.append([*box.tolist(), float(conf)])
                    people_lines.append(box_line(frame_number, box, conf))

            # 2. o rastreador dá um ID para cada pessoa
            tracks = tracker.update(detections, frame)  # id -> caixa
            for track_id, box in tracks.items():
                tracks_file.write(mot_line(frame_number, track_id, box))

            # 3. teu YOLO acha o jogador alvo (ou lê as caixas salvas). Fica com a caixa mais confiante.
            if saved is not None:
                target_boxes = saved_target.get(frame_number, [])  # cada uma: [x1, y1, x2, y2, conf]
            else:
                t0 = time.perf_counter()
                target_result = target_model.predict(frame, conf=TARGET_MIN_CONF, verbose=False)[0]
                target_seconds += time.perf_counter() - t0
                target_boxes = []
                for box, conf in zip(target_result.boxes.xyxy.cpu().numpy(), target_result.boxes.conf.cpu().numpy()):
                    target_boxes.append([*box.tolist(), float(conf)])
                    target_lines.append(box_line(frame_number, box, conf))
            target_box = None
            target_conf = 0.0
            if target_boxes:
                best = max(target_boxes, key=lambda b: b[4])
                target_box = best[:4]
                target_conf = best[4]

            lines = [(f"frame {frame_number}   |   {args.algorithm} {args.config}   |   "
                      f"{len(tracks)} people tracked", WHITE)]

            # 4. qual rastro é o jogador alvo
            if target_box is None:
                lines.append(("step 3: trained YOLO did NOT find the target player", GRAY))
                lines.append(("step 4: nothing to decide, keeps the previous ID", GRAY))
            elif not tracks:
                lines.append((f"step 3: trained YOLO found the target player (conf {target_conf:.2f})", YELLOW))
                lines.append(("step 4: no tracks to compare", GRAY))
            else:
                lines.append((f"step 3: trained YOLO found the target player (conf {target_conf:.2f})", YELLOW))
                best_id = max(tracks, key=lambda tid: iou(tracks[tid], target_box))
                best_iou = iou(tracks[best_id], target_box)
                if best_iou < MIN_IOU:
                    lines.append((f"step 4: best track ID {best_id} with IoU {best_iou:.2f} < {MIN_IOU} -> ignored", GRAY))
                elif target_id is None:
                    target_id = best_id
                    lines.append((f"step 4: IoU {best_iou:.2f} with ID {best_id} -> target player is ID {best_id}", GREEN))
                elif best_id == target_id:
                    lines.append((f"step 4: IoU {best_iou:.2f} with ID {best_id} -> confirmed", GREEN))
                else:
                    lines.append((f"step 4: IoU {best_iou:.2f} with ID {best_id} -> SWITCHED from {target_id} to {best_id}", RED))
                    target_id = best_id
                    id_changes += 1

            # 5. a caixa do jogador alvo é a do rastro com esse ID
            if target_id is not None and target_id in tracks:
                lines.append((f"step 5: target player = ID {target_id}", GREEN))
                trajectory_file.write(mot_line(frame_number, target_id, tracks[target_id]))
                frames_with_box += 1
            elif target_id is not None:
                lines.append((f"step 5: ID {target_id} is gone -> LOST in this frame", RED))
            else:
                lines.append(("step 5: target player not identified yet", GRAY))

            # Desenha (só se for mostrar ou gravar)
            if not args.no_window or writer is not None:
                for track_id, box in tracks.items():
                    if track_id != target_id:
                        draw_box(frame, box, GRAY, 1, str(track_id))
                if target_box is not None:
                    draw_box(frame, target_box, YELLOW, 2)
                if target_id is not None and target_id in tracks:
                    draw_box(frame, tracks[target_id], GREEN, 3, f"TARGET (ID {target_id})")
                draw_panel(frame, lines)
            if writer is not None:
                writer.write(frame)
            if not args.no_window:
                cv2.imshow(window, frame)

            frame_number += 1
            if frame_number % 900 == 0:
                speed = (frame_number - args.start_frame) / (time.perf_counter() - start_time)
                print(f"  frame {frame_number}/{end_frame} ({speed:.1f} frames/s)")

        if not args.no_window:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q ou Esc
                print("Parado antes do fim.")
                break
            if key == ord(" "):
                paused = not paused

    seconds = time.perf_counter() - start_time
    video.release()
    trajectory_file.close()
    tracks_file.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    n_frames = frame_number - args.start_frame

    # Tempo dos YOLOs nesta rodada: medido agora, ou a parte proporcional da vez em que rodaram
    if saved is None:
        save_detections(args.capture, args.start_frame, frame_number, people_lines, target_lines,
                        people_seconds, target_seconds)
        yolo_seconds = people_seconds + target_seconds
        tracking_seconds = seconds - yolo_seconds
    else:
        detected_frames = detections_info["end_frame"] - detections_info["start_frame"]
        yolo_seconds = (detections_info["people_seconds"] + detections_info["target_player_seconds"]) \
            * n_frames / detected_frames
        tracking_seconds = seconds

    run_info = {
        "capture": args.capture,
        "method": "mot_custom",
        "algorithm": args.algorithm,
        "config": args.config,
        "tracker_params": tracker_params,
        "detection_settings": detection_settings(args.capture),
        "detections_from_saved_files": saved is not None,
        "min_iou": MIN_IOU,
        "fps": fps,
        "start_frame": args.start_frame,
        "end_frame": frame_number,
        "frames": n_frames,
        "frames_with_box": frames_with_box,
        "id_changes": id_changes,
        # O FPS do sistema inteiro soma os YOLOs (mesmo quando vieram dos arquivos salvos) e o rastreamento
        "yolo_seconds": round(yolo_seconds, 1),
        "tracking_seconds": round(tracking_seconds, 1),
        "system_fps": round(n_frames / (yolo_seconds + tracking_seconds), 2) if n_frames else None,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPronto: {n_frames} frames em {seconds / 60:.1f} min, caixa em "
          f"{100 * frames_with_box / max(n_frames, 1):.1f}% dos frames, {id_changes} trocas de ID. "
          f"FPS do sistema: {run_info['system_fps']}")

    # Avaliação contra o gabarito, salva na mesma pasta
    evaluate(args.capture, run_dir)


if __name__ == "__main__":
    main()
