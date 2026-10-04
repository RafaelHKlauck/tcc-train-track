# Rastreamento do Araújo

Três versões, cada uma com SORT, DeepSORT e ByteTrack, nas duas capturas (`ang` = ângulo aberto, `broadcast`):

- **mot_classic**: o rastreador recebe só as caixas do teu detector do Araújo (a "abordagem SOT" do texto).
- **mot_custom**: o rastreador recebe todas as pessoas (YOLO pronto) e o teu detector diz qual ID é o Araújo.
- **mot_custom_team**: igual ao mot_custom, mas só entram as pessoas com a cor da camiseta do time.

## Estrutura

```
detection/                      tudo da detecção
  settings.py                   vídeos, modelos, parâmetros do YOLO e cores do time
  detect.py                     roda os YOLOs e salva as caixas (e roda sozinho quando um run_*.py precisa)
  team_color.py                 mede a cor da camiseta no tronco de cada pessoa
  check.py                      desenha imagens para conferir gabarito, detector e filtro de cor
  detect_ang.py                 ← roda os dois YOLOs do ângulo aberto de uma vez (opcional)
  detect_broadcast.py           ← roda a detecção do broadcast
  check_ang.py, check_broadcast.py

tracking/                       tudo do rastreamento
  settings.py                   gabarito (CVAT), parâmetros do MOT custom, filtro do time, avaliação
  ground_truth.py               junta as anotações do CVAT (min1, min2, ...) num gabarito só
  trackers.py                   SORT, DeepSORT e ByteTrack com o mesmo jeito de usar
  mot_classic.py                o laço da versão MOT clássico
  mot_custom.py                 o laço da versão MOT custom (o pseudocódigo)
  mot_custom_team.py            o mot_custom com o filtro do time ligado
  evaluation.py                 compara a trajetória com o gabarito
  compare_all.py                ← junta todas as rodadas numa tabela
  util.py                       funções pequenas
  ang/
    mot_classic/
      bytetrack/  run_default.py  run_custom.py
      deepsort/   run_default.py  run_custom.py
      sort/       run_default.py  run_custom.py
    mot_custom/       (igual)
    mot_custom_team/  (igual)
  broadcast/          (igual)

results/                        tudo o que os códigos geram
  detection/<captura>/people/   caixas do YOLO de pessoas (boxes.csv, info.json)
  detection/<captura>/araujo/   caixas do teu YOLO do Araújo (boxes.csv, info.json)
  detection/<captura>/check/    imagens do check
  tracking/<captura>/<versão>/<algoritmo>/<config>/
  tracking/summary.csv
```

Cada `run_default.py` / `run_custom.py` tem os parâmetros do rastreador escritos nele mesmo.
Para rodar só um pedaço do vídeo, passa `--end-frame` (ex.: `--end-frame 3600`). Se a rodada já existe, ele pergunta se é para
rodar de novo.

## Como rodar

Sempre na raiz do projeto (`cd "C:\Users\Rafael Hansen Klauck\codes\tcc\tcc-train-track"`).

**1. Rastrear + avaliar (cada combinação é um arquivo)**

```
python tracking/ang/mot_custom/bytetrack/run_default.py
python tracking/ang/mot_classic/bytetrack/run_default.py
...
```

Cada `run_*.py` roda sozinho o YOLO de que precisa, se ele ainda não rodou (ou não vai até o
`--end-frame` pedido). Depois disso as caixas ficam salvas e os outros `run_*.py` só leem:

| Versão | YOLO que roda antes, se precisar |
|---|---|
| mot_classic | só o do Araújo (o de pessoas roda só na avaliação, para separar troca de deriva) |
| mot_custom | pessoas + Araújo |
| mot_custom_team | pessoas (com a cor do time) + Araújo |

Para testar num pedaço curto, passa `--end-frame` (2 minutos = 3600 no ângulo aberto, 7200 no broadcast):

```
python tracking/ang/mot_custom/bytetrack/run_default.py --end-frame 3600
```
No fim de cada rodada aparece a avaliação contra o gabarito. O gabarito é montado sozinho na
primeira vez, a partir de `D:/tcc/angulo_aberto_track`.

Se quiser rodar os dois YOLOs de uma vez antes (por exemplo, o vídeo inteiro de noite):
`python detection/detect_ang.py` (também aceita `--end-frame`). Esse pergunta antes de refazer o que já existe.

**2. Conferir (opcional, recomendado na primeira vez)**

```
python detection/check_ang.py
```

Abre as imagens em `results/detection/ang/check/`. O verde (gabarito) tem que estar em cima do Araújo.

**3. Juntar tudo numa tabela**

```
python tracking/compare_all.py
```

## O que sai de cada rodada

Em `results/tracking/<captura>/<versão>/<algoritmo>/<config>/`:

- `trajectory.csv`: uma linha por frame com a caixa do Araújo (vazia quando o sistema não mostrou caixa)
- `tracks.csv`: tudo o que o rastreador devolveu (todas as pessoas)
- `run_info.json`: parâmetros, tempos e FPS
- `evaluation.json`: o resumo da avaliação
- `evaluation_frames.csv`: o rótulo de cada frame, para achar onde ele errou

## Avaliação

Cada frame recebe um rótulo, comparando a caixa do sistema com a do gabarito (IoU >= 0,5 = acerto):

| Rótulo | Quando |
|---|---|
| acerto (`correct`) | Araújo em campo e a caixa bate com o gabarito |
| perda (`lost`) | Araújo em campo e o sistema não mostrou caixa |
| troca (`switch`) | Araújo em campo e a caixa está em cima de outro jogador |
| deriva (`drift`) | Araújo em campo e a caixa está no fundo, sem ninguém embaixo |
| ausente (`absent`) | Araújo fora de cena e o sistema também não mostrou caixa (acerto) |
| alarme falso (`false_alarm`) | Araújo fora de cena e o sistema mostrou caixa |

Daí saem:

- **acerto**: % dos frames com o Araújo em campo em que o sistema acertou
- **quantas vezes perdeu** e **quantas vezes trocou**: cada bloco seguido de frames conta como 1 vez
- **recuperação**: uma falha começa no primeiro frame errado e termina quando o sistema volta a
  acertar. Frames com o Araújo fora de cena não contam no tempo. Mostra média, mediana e máximo
  em segundos, e quantas falhas nunca se recuperaram.

Para saber se uma caixa errada caiu em outro jogador (troca) ou no fundo (deriva), a avaliação usa
as caixas do YOLO de pessoas, inclusive na versão mot_classic.

## O FRAMES_TO_SWITCH (tracking/settings.py)

No mot_custom, para trocar o ID do Araújo para OUTRO rastro enquanto o rastro atual ainda está na
tela, o teu detector precisa apontar para o outro rastro em `FRAMES_TO_SWITCH` frames seguidos
(padrão 3). Isso evita o vai-e-volta quando o Araújo está colado em outro jogador. Se o rastro
atual sumiu, a troca é na hora.

## Segunda configuração dos rastreadores

Em cada `run_custom.py`: preenche `TRACKER_PARAMS` e `SOURCE` e muda `READY = True`.
