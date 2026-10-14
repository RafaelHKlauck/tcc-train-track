# SOT: rastreadores de objeto único (linha de base)

Roda rastreadores SOT no mesmo vídeo e com o mesmo gabarito do `tracking/run.py`, para comparar com o pipeline MOT.
Cada rastreador é inicializado com a caixa do gabarito no 1º frame em que o target player aparece.

| Arquivo | O que faz |
| --- | --- |
| `sot_config.py` | caminhos do vídeo e do gabarito de cada captura (preencher uma vez) |
| `sot_trackers.py` | os rastreadores, todos com `init(frame, caixa)` e `update(frame) -> (found, caixa, score)` |
| `run_sot.py` | roda um ou mais rastreadores numa captura e avalia no final |
| `evaluate_sot.py` | avaliação frame a frame e tabela com todas as rodadas |
| `download_models.py` | baixa os `.onnx` do DaSiamRPN, NanoTrack e VitTrack |

## Algoritmos

| Nome | Família | Artigo |
| --- | --- | --- |
| `csrt` | filtro de correlação | Lukežič et al., CVPR 2017 |
| `kcf` | filtro de correlação | Henriques et al., TPAMI 2015 |
| `mosse` | filtro de correlação | Bolme et al., CVPR 2010 |
| `mil` | aprendizado online (MIL) | Babenko, Yang e Belongie, CVPR 2009 |
| `medianflow` | fluxo óptico com detecção de falha | Kalal, Mikolajczyk e Matas, ICPR 2010 |
| `tld` | longo prazo (rastreia, detecta e aprende) | Kalal, Mikolajczyk e Matas, TPAMI 2012 |
| `dasiamrpn` | siamesa, treinada contra distratores | Zhu et al., ECCV 2018 |
| `nano` | siamesa leve | sem artigo próprio (NanoTrack) |
| `vit` | transformer leve | sem artigo próprio (VitTrack, OpenCV Zoo) |
| `dam4sam` | SAM 2.1 com memória contra distratores | Videnović, Lukežič e Kristan, CVPR 2025 |

Atalhos: `all-classic` (os seis primeiros) e `all-opencv` (os nove do OpenCV).

## Instalação

Os rastreadores clássicos estão no `opencv-contrib-python`, que briga com o `opencv-python` que o Ultralytics instala
(os dois disputam o `import cv2`). O mais simples é um ambiente separado, que não precisa de PyTorch:

```
cd "C:\Users\Rafael Hansen Klauck\codes\tcc\tcc-train-track"
python -m venv .venv-sot
.venv-sot\Scripts\activate
pip install -r tracking\SOT\requirements.txt
python tracking\SOT\download_models.py
```

Se preferir o mesmo ambiente do `run.py`: `pip uninstall -y opencv-python opencv-python-headless` e depois
`pip install opencv-contrib-python`. Atualizar o Ultralytics depois pode trazer o `opencv-python` de volta.

Os modelos ficam em `tracking/SOT/models/` (dá para pôr no `.gitignore`).

## Configurar

Em `sot_config.py`, preencha `video` e `gt` de `ang` e `broadcast` com os mesmos arquivos do `run.py`
(o gabarito unificado do `build_ground_truth.py`). Aceita MOT (`frame,id,x,y,w,h,...`, frame começando em 1)
ou CSV com `frame,x1,y1,x2,y2`. Também dá para passar `--video` e `--gt` na linha de comando.

## Rodar

Teste rápido primeiro (primeiros 2 min a 60 fps):

```
python tracking\SOT\run_sot.py --capture broadcast --algorithm csrt --protocol once --end-frame 7200 --save-video
```

Depois, o vídeo inteiro do gabarito:

```
python tracking\SOT\run_sot.py --capture broadcast --algorithm csrt dasiamrpn vit --protocol once reappear --no-window
python tracking\SOT\run_sot.py --capture ang --algorithm all-opencv --protocol once --no-window
python tracking\SOT\evaluate_sot.py --capture broadcast --all
```

Os frames de `--start-frame` e `--end-frame` são contados como no `gt.txt` (começam em 1). Sem eles, roda do 1º ao
último frame do gabarito. Na janela, `q` interrompe.

O MIL e o TLD são muito lentos em 1080p; para eles, vale rodar só um trecho e dizer isso no texto.

## Protocolos

- **`once`**: inicializa uma vez e nunca corrige. É o SOT puro, como na avaliação *one-pass* do OTB e do LaSOT.
- **`reappear`**: reinicializa pelo gabarito toda vez que o jogador volta à cena. Simula um redetector perfeito,
  então é o melhor que o SOT conseguiria. Se nem assim ele passar o pipeline MOT, o argumento fica forte.
- **`reset`**: como no VOT. Quando a caixa deixa de tocar no jogador (IoU = 0), espera 5 frames e reinicializa.
  O acerto fica inflado pelas correções; o número que importa são as falhas por minuto.

## Saídas

Em `results/<captura>/sot/<algoritmo>_<protocolo>/`:

- `trajectory.txt`: MOT (frame começa em 1), `frame,1,x,y,w,h,score,-1,-1,-1`, só os frames com caixa
- `frames.csv`: todos os frames, com a caixa crua, `found`, `score` e o evento (`init`, `reinit`, `reset`)
- `run_info.json`: parâmetros, artigo, versões, FPS e os frames de (re)inicialização
- `evaluation_sot.json` e `evaluation_frames_sot.csv`: a avaliação
- `video.mp4` (com `--save-video`): rastreador em verde, gabarito em magenta

`results/<captura>/sot/summary.csv` tem uma linha por rodada (gerado pelo `evaluate_sot.py --all`).

## Avaliação

O `evaluate_sot.py` classifica cada frame em acerto (IoU ≥ 0,5), perda, caixa errada, alarme falso e ausente certo,
e calcula a AUC da curva de sucesso e quanto tempo o rastreador seguiu o jogador até a 1ª falha
(falha = 0,5 s seguidos com o jogador em campo e sem acerto; ajustável com `--min-fail-seconds`).

Ele junta troca e deriva em "caixa errada", porque separar as duas exige as detecções de pessoas que estão no
`tracking/evaluate.py`. Para ter a avaliação completa e comparável com as 12 rodadas MOT, o `evaluate.py` precisa
ler o `trajectory.txt` de uma pasta qualquer (hoje ele procura em `mot_custom/<algoritmo>_<config>/`). Confira
também se o `trajectory.txt` do `run.py` tem o mesmo formato MOT descrito acima.

O DaSiamRPN, o NanoTrack e o VitTrack devolvem um score. `--score-threshold` faz frames com score baixo contarem
como "não achou" (o VitTrack já usa 0,2 por padrão). Se usar, escolha o limiar num trecho de ajuste, não nos
20 minutos de teste, e reavalie com `evaluate_sot.py --run <pasta> --score-threshold X`.

## DAM4SAM (opcional, estado da arte)

Precisa de GPU NVIDIA e funciona bem melhor no WSL2 (o `requirements.txt` dele pede `triton`, que não tem versão
para Windows). Num ambiente próprio:

```
git clone https://github.com/jovanavidenovic/DAM4SAM.git
cd DAM4SAM
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd checkpoints && ./download_ckpts.sh
```

Depois, `DAM4SAM_REPO` no `sot_config.py` (ou `--dam4sam-repo`) e `--algorithm dam4sam`. O original guarda a
memória de todos os frames na GPU, o que estoura em 20 minutos de vídeo; o `sot_trackers.py` poda os frames antigos
que o modelo não consulta mais. Teste antes num trecho curto (`--end-frame 3000`), porque essa parte não foi
rodada com GPU.
