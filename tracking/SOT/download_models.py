"""
Baixa os modelos ONNX dos rastreadores com rede do OpenCV para tracking/SOT/models/.
Os endereços e SHA1 são os mesmos que o próprio OpenCV usa nos testes (opencv_extra/download_models.py).

Uso:
    python tracking/SOT/download_models.py            # todos
    python tracking/SOT/download_models.py vit nano   # só alguns
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sot_config  # noqa: E402

MODELS = {
    "vit": [
        ("object_tracking_vittrack_2023sep.onnx", "50008bb4f6a27b1aa940ad886b1bd1936ac4ed3e", [
            "https://github.com/opencv/opencv_zoo/raw/fef72f8fa7c52eaf116d3df358d24e6e959ada0e/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
        ]),
    ],
    "nano": [
        ("nanotrack_backbone_sim_v2.onnx", "6e773a364457b78574f9f63a23b0659ee8646f8f", [
            "https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrackV2/models/nanotrack_backbone_sim_v2.onnx",
            "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx",
        ]),
        ("nanotrack_head_sim_v2.onnx", "39f168489671700cf739e402dfc67d41ce648aef", [
            "https://raw.githubusercontent.com/zihaomu/opencv_extra_data_backup/main/NanoTrackV2/models/nanotrack_head_sim_v2.onnx",
            "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx",
        ]),
    ],
    "dasiamrpn": [
        ("dasiamrpn_model.onnx", "91b774fce7df4c0e4918469f0f482d9a27d0e2d4",
         ["https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=1"]),
        ("dasiamrpn_kernel_r1.onnx", "bb64620a54348657133eb28be2d3a2a8c76b84b3",
         ["https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=1"]),
        ("dasiamrpn_kernel_cls1.onnx", "e9ccd270ce8059bdf7ed0d1845c03ef4a951ee0f",
         ["https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=1"]),
    ],
}


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(filename, expected_sha1, urls, dest_dir):
    dest = dest_dir / filename
    if dest.exists() and sha1(dest) == expected_sha1:
        print(f"  {filename}: já existe")
        return True
    for url in urls:
        try:
            print(f"  {filename}: baixando de {url.split('/')[2]} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
                out.write(r.read())
        except Exception as e:  # tenta o próximo endereço
            print(f"    falhou: {e}")
            continue
        got = sha1(dest)
        if got == expected_sha1:
            print("    ok (SHA1 confere)")
            return True
        print(f"    SHA1 diferente do esperado ({got}); tentando o próximo endereço")
    if dest.exists():
        print(f"  {filename}: AVISO, ficou uma versão com SHA1 diferente do esperado. Anote isso no run_info.")
        return True
    print(f"  {filename}: não consegui baixar. Baixe manualmente e coloque em {dest_dir}")
    return False


def main():
    names = sys.argv[1:] or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"Desconhecido: {unknown}. Opções: {', '.join(MODELS)}")
    dest_dir = Path(sot_config.MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in names:
        print(f"{name}:")
        for filename, expected, urls in MODELS[name]:
            ok &= download(filename, expected, urls, dest_dir)
    print("Pronto." if ok else "Alguns arquivos faltaram (ver acima).")


if __name__ == "__main__":
    main()
