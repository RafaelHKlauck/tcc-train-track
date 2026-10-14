"""
Avaliação das rodadas de SOT contra o gabarito, frame a frame.

Classes de cada frame (no CSV ficam em inglês):
    hit          jogador no frame e caixa com IoU >= 0,5                   (acerto)
    miss         jogador no frame e nenhuma caixa                         (perda)
    wrong        jogador no frame, mas a caixa está em outro lugar         (troca ou deriva)
    false_alarm  jogador fora de cena e o rastreador mostrou caixa         (alarme falso)
    absent_ok    jogador fora de cena e nenhuma caixa                      (ausente certo)

"wrong" junta troca e deriva: separar as duas exige as detecções de pessoas, que estão no
tracking/evaluate.py. Para ter a separação, rode o evaluate.py no trajectory.txt da rodada (ver README).

Também calcula o que é específico de SOT:
    - AUC da curva de sucesso (OTB, 21 limiares) e IoU médio nos frames com o jogador
    - quanto tempo o rastreador seguiu o jogador depois de cada (re)inicialização até a primeira falha
    - no protocolo reset: falhas por minuto com o jogador em campo (robustez, como no VOT)

Uso:
    python tracking/SOT/evaluate_sot.py --run results/broadcast/sot/csrt_once
    python tracking/SOT/evaluate_sot.py --capture broadcast --all       # tabela com todas as rodadas
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

import sot_config

IOU_THRESHOLD = 0.5
SUCCESS_THRESHOLDS = np.linspace(0, 1, 21)
LABELS_PT = {"hit": "acerto", "miss": "perda", "wrong": "caixa errada (troca/deriva)",
             "false_alarm": "alarme falso", "absent_ok": "ausente certo"}


# ------------------------------------------------------------------ leitura

def load_ground_truth(path):
    """Gabarito -> {frame: (x, y, w, h)}, frame começando em 1 (como no MOT). Frame fora = jogador fora de cena."""
    path = Path(path)
    gt, duplicated = {}, 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        if any(c.isalpha() for c in first):  # CSV com cabeçalho (frame, x1, y1, x2, y2, ...)
            for row in csv.DictReader(f):
                vis = row.get("visible", row.get("visivel", "1"))
                if str(vis).strip() in ("0", "") or not row.get("x1"):
                    continue
                x1, y1, x2, y2 = (float(row[k]) for k in ("x1", "y1", "x2", "y2"))
                fr = int(float(row["frame"]))
                duplicated += fr in gt
                gt.setdefault(fr, (x1, y1, x2 - x1, y2 - y1))
        else:  # MOT: frame,id,x,y,w,h,not_ignored,class,visibility
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                vals = [float(p) for p in parts]
                if len(vals) >= 7 and vals[6] == 0:  # linha marcada como ignorada
                    continue
                fr = int(vals[0])
                duplicated += fr in gt
                gt.setdefault(fr, tuple(vals[2:6]))
    if not gt:
        raise ValueError(f"Nenhuma caixa lida do gabarito {path}")
    if duplicated:
        print(f"Aviso: {duplicated} frames com mais de uma caixa no gabarito; usei a primeira de cada.")
    return gt


def load_frames(run_dir):
    """frames.csv da rodada -> lista de dicts por frame."""
    rows = []
    with open(Path(run_dir) / "frames.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            box = None
            if r["x"] != "":
                box = (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            rows.append({"frame": int(r["frame"]), "found": r["found"] == "1", "box": box,
                         "score": float(r["score"]) if r["score"] != "" else None, "event": r["event"]})
    return rows


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def is_shown(row, score_threshold):
    """O sistema mostrou caixa neste frame? (rastreador achou e, se houver limiar, o score passou)"""
    if not row["found"] or row["box"] is None:
        return False
    return score_threshold is None or row["score"] is None or row["score"] >= score_threshold


# ------------------------------------------------------------------ avaliação

def label_frames(rows, gt, score_threshold=None):
    out = []
    for r in rows:
        g = gt.get(r["frame"])
        shown = is_shown(r, score_threshold)
        iou = iou_xywh(r["box"], g) if (shown and g is not None) else (0.0 if g is not None else None)
        if g is not None:
            label = "miss" if not shown else ("hit" if iou >= IOU_THRESHOLD else "wrong")
        else:
            label = "false_alarm" if shown else "absent_ok"
        out.append({**r, "gt": g, "shown": shown, "iou": iou, "label": label})
    return out


def time_until_failure(labeled, fps, min_fail_frames, reset_failures=None):
    """
    Para cada trecho entre uma (re)inicialização e a próxima: quantos segundos o rastreador seguiu o
    jogador até a primeira falha. Falha = min_fail_frames frames seguidos COM o jogador em campo e sem
    acerto (perda ou caixa errada). Frames sem o jogador no gabarito não contam nem como falha nem como acerto.
    No protocolo reset, a falha é a que o próprio protocolo registrou (IoU = 0), que dispara o reset.
    """
    segments, current = [], None
    for i, r in enumerate(labeled):
        if r["event"] in ("init", "reinit", "reset"):
            if current:
                segments.append(current)
            current = {"start": r["frame"], "failed_at": None, "end": r["frame"], "run": 0, "run_start": None}
        if current is None:
            continue
        current["end"] = r["frame"]
        if current["failed_at"] is not None or r["gt"] is None:
            continue
        if r["label"] in ("miss", "wrong"):
            if current["run"] == 0:
                current["run_start"] = r["frame"]
            current["run"] += 1
            if current["run"] >= min_fail_frames:
                current["failed_at"] = current["run_start"]
        else:
            current["run"] = 0
    if current:
        segments.append(current)

    if reset_failures is not None:  # protocolo reset: usa as falhas registradas na rodada
        for s in segments:
            inside = [f for f in reset_failures if s["start"] <= f <= s["end"]]
            s["failed_at"] = inside[0] if inside else None

    result = []
    for s in segments:
        failed = s["failed_at"] is not None
        last = s["failed_at"] if failed else s["end"] + 1
        result.append({"init_frame": s["start"], "failed": failed,
                       "failure_frame": s["failed_at"], "seconds": round((last - s["start"]) / fps, 2)})
    return result


def evaluate_run(run_dir, gt=None, score_threshold="from_run", min_fail_seconds=0.5, write=True):
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    if gt is None:
        gt = load_ground_truth(info["gt"])
    if score_threshold == "from_run":
        score_threshold = info.get("score_threshold")
    fps = info["fps"]
    rows = load_frames(run_dir)
    labeled = label_frames(rows, gt, score_threshold)

    counts = {k: 0 for k in LABELS_PT}
    for r in labeled:
        counts[r["label"]] += 1
    n = len(labeled)
    with_target = counts["hit"] + counts["miss"] + counts["wrong"]
    without_target = counts["false_alarm"] + counts["absent_ok"]

    ious = np.array([r["iou"] for r in labeled if r["gt"] is not None], dtype=float)
    success = [float((ious > t).mean()) if len(ious) else 0.0 for t in SUCCESS_THRESHOLDS]

    min_fail_frames = max(1, int(round(min_fail_seconds * fps)))
    reset_failures = info.get("failures") if info["protocol"] == "reset" else None
    segments = time_until_failure(labeled, fps, min_fail_frames, reset_failures)
    failed = [s["seconds"] for s in segments if s["failed"]]

    pct = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
    summary = {
        "capture": info["capture"], "algorithm": info["algorithm"], "protocol": info["protocol"],
        "frames": n, "frames_with_target": with_target, "frames_without_target": without_target,
        "counts": counts,
        "hit_pct_of_target_frames": pct(counts["hit"], with_target),
        "miss_pct_of_target_frames": pct(counts["miss"], with_target),
        "wrong_pct_of_target_frames": pct(counts["wrong"], with_target),
        "false_alarm_pct_of_absent_frames": pct(counts["false_alarm"], without_target),
        "overall_correct_pct": pct(counts["hit"] + counts["absent_ok"], n),
        "auc_success": round(float(np.mean(success)), 4),
        "mean_iou_target_frames": round(float(ious.mean()), 4) if len(ious) else None,
        "success_curve": {f"{t:.2f}": round(s, 4) for t, s in zip(SUCCESS_THRESHOLDS, success)},
        "inits": len(segments),
        "segments_failed": len(failed),
        "segments_never_failed": len(segments) - len(failed),
        "seconds_until_first_failure": segments[0]["seconds"] if segments else None,
        "first_segment_failed": segments[0]["failed"] if segments else None,
        "median_seconds_until_failure": round(statistics.median(failed), 2) if failed else None,
        "mean_seconds_until_failure": round(statistics.mean(failed), 2) if failed else None,
        "min_fail_seconds": min_fail_seconds,
        "score_threshold": score_threshold,
        "iou_threshold": IOU_THRESHOLD,
        "fps_tracker": info.get("fps_tracker"),
    }
    if info["protocol"] == "reset":
        minutes = with_target / fps / 60
        summary["resets"] = len(info.get("failures", []))
        summary["failures_per_minute_with_target"] = round(summary["resets"] / minutes, 2) if minutes else None

    if write:
        (run_dir / "evaluation_sot.json").write_text(
            json.dumps({**summary, "segments": segments}, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(run_dir / "evaluation_frames_sot.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "gt_present", "shown", "iou", "label", "event"])
            for r in labeled:
                w.writerow([r["frame"], int(r["gt"] is not None), int(r["shown"]),
                            "" if r["iou"] is None else round(r["iou"], 4), r["label"], r["event"]])
    return summary


def print_summary(s):
    print(f"\n=== {s['capture']} | {s['algorithm']} | protocolo {s['protocol']} ===")
    print(f"Frames avaliados: {s['frames']} (com o jogador: {s['frames_with_target']}, sem: {s['frames_without_target']})")
    print(f"  acerto (frames com o jogador):        {s['hit_pct_of_target_frames']}%")
    print(f"  perda:                                {s['miss_pct_of_target_frames']}%")
    print(f"  caixa errada (troca ou deriva):       {s['wrong_pct_of_target_frames']}%")
    print(f"  alarme falso (frames sem o jogador):  {s['false_alarm_pct_of_absent_frames']}%")
    print(f"  acerto geral (acerto + ausente certo): {s['overall_correct_pct']}%")
    print(f"  AUC da curva de sucesso: {s['auc_success']} | IoU médio: {s['mean_iou_target_frames']}")
    print(f"  (re)inicializações: {s['inits']} | trechos que falharam: {s['segments_failed']}")
    if s["first_segment_failed"]:
        print(f"  seguiu o jogador por {s['seconds_until_first_failure']} s até a 1ª falha")
    elif s["first_segment_failed"] is not None:
        print(f"  não falhou no 1º trecho ({s['seconds_until_first_failure']} s, até a próxima inicialização ou o fim)")
    if s["segments_failed"]:
        print(f"  tempo até falhar, entre os trechos que falharam: mediana {s['median_seconds_until_failure']} s, "
              f"média {s['mean_seconds_until_failure']} s")
    if "resets" in s:
        print(f"  resets: {s['resets']} ({s['failures_per_minute_with_target']} por minuto com o jogador em campo)")


def summarize_capture(capture):
    base = sot_config.RESULTS_DIR / capture / "sot"
    runs = sorted(p.parent for p in base.glob("*/run_info.json"))
    if not runs:
        print(f"Nenhuma rodada em {base}")
        return
    cols = ["algorithm", "protocol", "hit_pct_of_target_frames", "miss_pct_of_target_frames",
            "wrong_pct_of_target_frames", "false_alarm_pct_of_absent_frames", "overall_correct_pct",
            "auc_success", "inits", "seconds_until_first_failure", "first_segment_failed", "median_seconds_until_failure",
            "failures_per_minute_with_target", "fps_tracker"]
    table, gt_cache = [], {}
    for run in runs:
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        if info["gt"] not in gt_cache:
            gt_cache[info["gt"]] = load_ground_truth(info["gt"])
        s = evaluate_run(run, gt=gt_cache[info["gt"]])
        table.append({c: s.get(c, "") for c in cols})
    with open(base / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    head = ["algoritmo", "protocolo", "acerto%", "perda%", "errada%", "alarme%", "geral%", "AUC", "inits", "1ª falha s", "fps"]
    print("  ".join(f"{h:>10}" for h in head))
    for t in table:
        vals = [t["algorithm"], t["protocol"], t["hit_pct_of_target_frames"], t["miss_pct_of_target_frames"],
                t["wrong_pct_of_target_frames"], t["false_alarm_pct_of_absent_frames"], t["overall_correct_pct"],
                t["auc_success"], t["inits"],
                f"{t['seconds_until_first_failure']}{'' if t['first_segment_failed'] else '+'}", t["fps_tracker"]]
        print("  ".join(f"{str(v):>10}" for v in vals))
    print("\n'+' na coluna da 1ª falha: o 1º trecho não falhou (o valor é só até a próxima inicialização ou o fim).")
    print(f"Tabela salva em {base / 'summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description="Avalia rodadas de SOT contra o gabarito.")
    ap.add_argument("--run", help="pasta da rodada (results/<captura>/sot/<algoritmo>_<protocolo>)")
    ap.add_argument("--capture", choices=list(sot_config.CAPTURES), help="com --all: avalia todas as rodadas da captura")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gt", help="outro gabarito (padrão: o que a rodada usou)")
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="refaz a avaliação com outro limiar de score (só vale para dasiamrpn, nano, vit)")
    ap.add_argument("--min-fail-seconds", type=float, default=0.5,
                    help="quantos segundos seguidos sem acerto contam como falha (padrão 0,5 s)")
    a = ap.parse_args()
    if a.all:
        if not a.capture:
            ap.error("--all precisa de --capture")
        summarize_capture(a.capture)
        return
    if not a.run:
        ap.error("informe --run ou --capture X --all")
    gt = load_ground_truth(a.gt) if a.gt else None
    thr = a.score_threshold if a.score_threshold is not None else "from_run"
    print_summary(evaluate_run(a.run, gt=gt, score_threshold=thr, min_fail_seconds=a.min_fail_seconds))


if __name__ == "__main__":
    main()
