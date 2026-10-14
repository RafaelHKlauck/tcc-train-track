"""
Roda rastreadores de objeto único (SOT) no vídeo de uma captura, inicializando com o gabarito.

Protocolos (--protocol):
  once      Inicializa uma vez, no 1º frame em que o target player aparece, e nunca mais corrige.
            É o SOT puro, como na avaliação one-pass do OTB/LaSOT.
  reappear  Reinicializa pelo gabarito sempre que o jogador volta à cena depois de sumir.
            Simula um redetector perfeito: é o melhor que o SOT conseguiria com uma redetecção ideal.
  reset     Como no VOT: quando a caixa deixa de tocar no jogador (IoU = 0), espera --reset-delay
            frames e reinicializa pelo gabarito. O número que interessa aqui são as falhas por minuto,
            não o acerto (que fica inflado pelas correções).

Saídas em results/<captura>/sot/<algoritmo>_<protocolo>/:
  trajectory.txt           MOT (frame começa em 1): frame,1,x,y,w,h,score,-1,-1,-1, só frames com caixa
  frames.csv               um frame por linha: caixa crua, found, score e evento (init/reinit/reset)
  run_info.json            parâmetros, versões, tempos, FPS e frames de (re)inicialização
  evaluation_sot.json      resumo da avaliação (evaluate_sot.py)
  evaluation_frames_sot.csv  classe de cada frame
  video.mp4                com --save-video (rastreador em verde, gabarito em magenta)

Exemplos:
  python tracking/SOT/run_sot.py --capture broadcast --algorithm csrt --protocol once --no-window
  python tracking/SOT/run_sot.py --capture ang --algorithm csrt kcf vit --protocol once reappear --no-window
  python tracking/SOT/run_sot.py --capture broadcast --algorithm all-opencv --protocol once --no-window
"""
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sot_config  # noqa: E402
from evaluate_sot import evaluate_run, iou_xywh, load_ground_truth, print_summary  # noqa: E402
from sot_trackers import ALGORITHMS, OPENCV_CLASSIC, OPENCV_DNN, create_tracker  # noqa: E402

PROTOCOLS = ["once", "reappear", "reset"]
COLOR_TRACKER = (0, 255, 0)   # verde
COLOR_GT = (255, 0, 255)      # magenta, igual ao export_examples.py
LOG_EVERY_S = 10


def parse_args():
    ap = argparse.ArgumentParser(description="Rastreadores SOT inicializados pelo gabarito.")
    ap.add_argument("--capture", required=True, choices=list(sot_config.CAPTURES))
    ap.add_argument("--algorithm", nargs="+", required=True,
                    help=f"um ou mais de: {', '.join(ALGORITHMS)}; ou all-opencv / all-classic")
    ap.add_argument("--protocol", nargs="+", default=["once"], choices=PROTOCOLS)
    ap.add_argument("--start-frame", type=int, default=None,
                    help="primeiro frame, contado como no gt.txt (começa em 1). Padrão: 1º frame do gabarito")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="último frame (inclusive). Padrão: último frame do gabarito")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="só dasiamrpn/nano/vit: abaixo disto conta como 'não achou'. Padrão: nenhum")
    ap.add_argument("--reset-delay", type=int, default=5, help="protocolo reset: frames até reinicializar (VOT usa 5)")
    ap.add_argument("--video", help="sobrescreve o vídeo do sot_config.py")
    ap.add_argument("--gt", help="sobrescreve o gabarito do sot_config.py")
    ap.add_argument("--models-dir", default=None, help="pasta dos .onnx (padrão: tracking/SOT/models)")
    ap.add_argument("--dam4sam-repo", default=None, help="pasta do repositório do DAM4SAM")
    ap.add_argument("--no-window", action="store_true", help="não abre a janela de visualização")
    ap.add_argument("--save-video", action="store_true", help="salva video.mp4 com as caixas")
    ap.add_argument("--no-eval", action="store_true", help="não roda a avaliação no final")
    return ap.parse_args()


def expand_algorithms(names):
    out = []
    for n in names:
        n = n.lower()
        if n == "all-opencv":
            out += OPENCV_CLASSIC + OPENCV_DNN
        elif n == "all-classic":
            out += OPENCV_CLASSIC
        elif n in ALGORITHMS:
            out.append(n)
        else:
            raise SystemExit(f"Algoritmo desconhecido: {n}. Opções: {', '.join(ALGORITHMS)}, all-opencv, all-classic")
    return list(dict.fromkeys(out))  # sem repetir, mantendo a ordem


def resolve_paths(args):
    cfg = sot_config.CAPTURES[args.capture]
    video = args.video or cfg["video"]
    gt = args.gt or cfg["gt"]
    if not video or not gt:
        raise SystemExit(f"Defina o vídeo e o gabarito da captura '{args.capture}' em tracking/SOT/sot_config.py "
                         f"(ou use --video e --gt).")
    video, gt = Path(video), Path(gt)
    for p in (video, gt):
        if not p.exists():
            raise SystemExit(f"Arquivo não encontrado: {p}")
    return video, gt


def open_video_at(video, start_frame):
    """Abre o vídeo e posiciona no frame start_frame (contado a partir de 1). Lê em sequência para não errar o frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo {video}")
    for _ in range(start_frame - 1):
        if not cap.grab():
            raise SystemExit(f"O vídeo acabou antes do frame {start_frame}")
    return cap


def fmt(v, nd=1):
    return "" if v is None else f"{v:.{nd}f}"


def draw(frame, f, box, gt_box, name, protocol, event):
    vis = frame.copy()
    if gt_box is not None:
        x, y, w, h = (int(round(v)) for v in gt_box)
        cv2.rectangle(vis, (x, y), (x + w, y + h), COLOR_GT, 2)
    if box is not None:
        x, y, w, h = (int(round(v)) for v in box)
        cv2.rectangle(vis, (x, y), (x + w, y + h), COLOR_TRACKER, 2)
    scale = max(0.5, vis.shape[1] / 1600)
    text = f"frame {f} | {name.upper()} ({protocol}) | green: tracker, magenta: ground truth"
    if event:
        text += f" | {event}"
    cv2.putText(vis, text, (15, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(vis, text, (15, int(35 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return vis


def run_one(args, name, protocol, video, gt_path, gt, start, end):
    label = f"{args.capture}/{name}/{protocol}"
    out_dir = sot_config.RESULTS_DIR / args.capture / "sot" / f"{name}_{protocol}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = create_tracker(name, args.models_dir, args.dam4sam_repo)
    cap = open_video_at(video, start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        raise SystemExit("O vídeo não informou o FPS.")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    show = not args.no_window
    if show:
        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(label, min(1280, width), int(min(1280, width) * height / width))

    print(f"\n[{label}] frames {start} a {end} ({(end - start + 1) / fps / 60:.1f} min a {fps:.2f} fps)")
    initialized, prev_visible, reset_at = False, False, None
    inits, failures, rows = [], [], []
    tracker_time, shown_count = 0.0, 0
    t0 = last_log = time.perf_counter()

    for f in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"[{label}] Aviso: o vídeo acabou no frame {f - 1}, antes do fim pedido ({end}).")
            end = f - 1
            break
        g = gt.get(f)
        visible = g is not None

        # Decide se (re)inicializa pelo gabarito neste frame
        event = ""
        if visible and not initialized:
            event = "init"
        elif visible and protocol == "reappear" and not prev_visible:
            event = "reinit"
        elif visible and protocol == "reset" and reset_at is not None and f >= reset_at:
            event = "reset"

        found, box, score = False, None, None
        t = time.perf_counter()
        if event:
            tracker.init(frame, g)
            initialized, reset_at = True, None
            found, box = True, tuple(float(v) for v in g)
            inits.append({"frame": f, "type": event})
        elif initialized and reset_at is None:
            found, box, score = tracker.update(frame)
        tracker_time += time.perf_counter() - t

        shown = found and box is not None and (
            args.score_threshold is None or score is None or score >= args.score_threshold)
        shown_count += shown

        # Protocolo reset: falha quando a caixa não toca no jogador (ou some com ele em cena)
        if protocol == "reset" and visible and not event and reset_at is None and initialized:
            if not shown or iou_xywh(box, g) <= 0:
                failures.append(f)
                reset_at = f + args.reset_delay

        rows.append([f, int(found), *(fmt(v) for v in (box or (None,) * 4)), fmt(score, 4), event])
        prev_visible = visible

        if writer is not None or show:
            vis = draw(frame, f, box if shown else None, g, name, protocol, event)
            if writer is not None:
                writer.write(vis)
            if show:
                cv2.imshow(label, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"[{label}] Interrompido pelo usuário no frame {f}.")
                    end = f
                    break

        now = time.perf_counter()
        if now - last_log >= LOG_EVERY_S:
            done = f - start + 1
            print(f"[{label}] frame {f}/{end} ({100 * done / (end - start + 1):.1f}%) | "
                  f"{done / (now - t0):.1f} fps | caixa em {100 * shown_count / done:.1f}% dos frames | "
                  f"(re)inicializações: {len(inits)}")
            last_log = now

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyWindow(label)
    wall = time.perf_counter() - t0
    n = len(rows)

    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "found", "x", "y", "w", "h", "score", "event"])
        w.writerows(rows)

    with open(out_dir / "trajectory.txt", "w", encoding="utf-8") as fh:
        for f_, found_, x, y, bw, bh, sc, _ in rows:
            if not found_ or x == "":
                continue
            if args.score_threshold is not None and sc != "" and float(sc) < args.score_threshold:
                continue
            fh.write(f"{f_},1,{x},{y},{bw},{bh},{sc or '1'},-1,-1,-1\n")

    info = {
        "capture": args.capture, "algorithm": name, "protocol": protocol,
        "family": ALGORITHMS[name]["family"], "paper": ALGORITHMS[name]["paper"],
        "tracker_params": tracker.params,
        "video": str(video), "gt": str(gt_path), "fps": fps, "resolution": [width, height],
        "start_frame": start, "end_frame": end, "frames": n,
        # start_frame/end_frame aqui já estão no índice do MOT (começa em 1);
        # o tracking/run.py grava start_frame no índice do OpenCV (começa em 0).
        # O tracking/evaluate.py lê esta chave para não errar por um frame.
        "frame_base": 1,
        "score_threshold": args.score_threshold, "reset_delay": args.reset_delay if protocol == "reset" else None,
        "inits": inits, "failures": failures,
        "tracker_time_s": round(tracker_time, 2), "fps_tracker": round(n / tracker_time, 2) if tracker_time else None,
        "wall_time_s": round(wall, 2), "fps_wall": round(n / wall, 2) if wall else None,
        "frames_with_box_pct": round(100 * shown_count / n, 2) if n else None,
        "opencv": cv2.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "command": " ".join(sys.argv), "date": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{label}] Pronto em {wall / 60:.1f} min ({info['fps_tracker']} fps só do rastreador). "
          f"Saída em {out_dir}")

    if not args.no_eval:
        print_summary(evaluate_run(out_dir, gt=gt))
    return out_dir


def main():
    args = parse_args()
    algorithms = expand_algorithms(args.algorithm)
    video, gt_path = resolve_paths(args)
    gt = load_ground_truth(gt_path)
    start = args.start_frame or min(gt)
    end = args.end_frame or max(gt)
    if end < start:
        raise SystemExit(f"--end-frame ({end}) menor que --start-frame ({start}).")
    if not any(start <= f <= end for f in gt):
        raise SystemExit("O jogador não aparece no gabarito dentro desse trecho: não há como inicializar.")
    print(f"Gabarito: {len(gt)} frames com o jogador (frames {min(gt)} a {max(gt)}). Trecho pedido: {start} a {end}.")

    for name in algorithms:
        for protocol in args.protocol:
            try:
                run_one(args, name, protocol, video, gt_path, gt, start, end)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[{args.capture}/{name}/{protocol}] Pulado: {e}")


if __name__ == "__main__":
    main()
