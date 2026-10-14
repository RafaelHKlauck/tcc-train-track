"""
Junta o gabarito do CVAT (export MOT 1.1, uma pasta por minuto) num arquivo só.

Entrada:  D:/tcc/angulo_aberto_track/min1/gt/gt.txt, min2/gt/gt.txt, ... min20/gt/gt.txt
          cada linha: frame, id, x, y, largura, altura, não_ignorado, classe, visibilidade
          (em cada pasta o frame recomeça do 1)

Saída:    D:/tcc/ang_unified/gt.txt   (mesmo formato MOT, com o frame do vídeo, começando em 1)

O frame do vídeo é:  (minuto - 1) * fps * 60 + frame_do_gt
Frame sem linha = jogador fora de cena.

O run.py chama isso sozinho se o gabarito ainda não existe. Para rodar à mão (na raiz do projeto):
    python tracking/build_ground_truth.py
    python tracking/build_ground_truth.py --capture broadcast
"""
import argparse
import re
from pathlib import Path

CAPTURES = {
    "ang": {"annotations": Path("D:/tcc/angulo_aberto_track"), "output": Path("D:/tcc/ang_unified"), "fps": 30},
    "broadcast": {"annotations": Path("D:/tcc/broadcast_track"), "output": Path("D:/tcc/broadcast_unified"), "fps": 60},
}


def build_ground_truth(capture):
    """Cria D:/tcc/<captura>_unified/gt.txt e devolve o caminho dele."""
    info = CAPTURES[capture]
    frames_per_minute = info["fps"] * 60

    # Acha as pastas min1, min2, ... e ordena pelo número (min2 antes de min10)
    minutes = []
    for folder in info["annotations"].iterdir():
        match = re.fullmatch(r"min(\d+)", folder.name)
        if match and (folder / "gt" / "gt.txt").exists():
            minutes.append((int(match.group(1)), folder / "gt" / "gt.txt"))
    minutes.sort()
    if not minutes:
        raise FileNotFoundError(f"Nenhuma pasta min<N>/gt/gt.txt em {info['annotations']}")

    output_lines = []
    for minute, gt_file in minutes:
        offset = (minute - 1) * frames_per_minute
        for line in gt_file.read_text(encoding="utf-8").splitlines():
            values = line.strip().split(",")
            if len(values) < 6:
                continue
            values[0] = str(int(values[0]) + offset)  # frame do minuto -> frame do vídeo
            values[1] = "1"                           # um jogador só: ID 1
            output_lines.append(values)

    output_lines.sort(key=lambda values: int(values[0]))
    info["output"].mkdir(parents=True, exist_ok=True)
    out_file = info["output"] / "gt.txt"
    out_file.write_text("\n".join(",".join(values) for values in output_lines) + "\n", encoding="utf-8")

    total_frames = len(minutes) * frames_per_minute
    print(f"Gabarito: {len(minutes)} minutos ({total_frames} frames), jogador visível em {len(output_lines)} frames "
          f"({100 * len(output_lines) / total_frames:.1f}%). Salvo em {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["ang", "broadcast"], default="ang")
    args = parser.parse_args()
    build_ground_truth(args.capture)


if __name__ == "__main__":
    main()
