# tcc-train-track

Harness de treino dos detectores YOLO do TCC *"Análise do desempenho de
rastreamento automático de jogadores em partidas de futebol sob diferentes
condições de captura de vídeo"*.

Monta os conjuntos de treino em diferentes volumes de dados anotados
(1, 5, 10, 20 e 40 minutos), treina por 50 épocas e guarda os pesos nos
checkpoints de 10, 25 e 50 épocas para avaliação posterior.

## Instalação

```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dados de origem

Exportação do CVAT no formato *YOLO 1.1*, uma pasta por minuto de vídeo:

```
D:\tcc\angulo_aberto\
  min1\
    obj_train_data\frame_001800.png
                   frame_001800.txt
                   ...
    train.txt, obj.names, obj.data
  min2\ ...
```

A numeração dos frames é global e contínua dentro de cada condição, e cada
pasta `min<N>` cobre exatamente 1 minuto de vídeo.

### As duas condições de captura

`--condition ang` (padrão) ou `--condition broadcast`. As duas fontes **não têm
a mesma taxa de quadros**, e é por isso que a geometria é declarada por condição
em [`config.py`](config.py) em vez de ser uma constante global:

| | `ang` (ângulo aberto) | `broadcast` |
|---|---|---|
| Raiz | `D:\tcc\angulo_aberto` | `D:\tcc\broadcast` |
| Resolução | 1920×822 | 1920×1080 |
| FPS | 30 | 60 |
| Frames por pasta `min` | 1.800 | 3.600 |
| `min1` começa no frame | 1800 | 0 |
| Stride padrão | 3 | 6 |
| Caixa mediana | ~24×69 px | ~170×468 px |

As raízes podem ser sobrescritas por variável de ambiente (`TCC_DATA_ANG`,
`TCC_DATA_BROADCAST`).

A diferença de escala das caixas (~48× em área) é grande porque o broadcast é um
plano fechado. Isso é justamente o que o trabalho mede, mas vale registrar no
texto que a comparação não isola só "broadcast vs wide-angle" — tem uma
diferença de zoom junto.

## Uso

```bash
python train.py 5                    # monta o cenário de 5 min e treina 50 épocas
python train.py 10 --aug scale       # segunda configuração de aumento (seção 4.5)
python train.py 40 --stride 1        # sem subamostragem de frames
python train.py 5 --resume           # retoma um treino interrompido
python prepare.py 20 --dry-run       # só valida e mostra as contagens
```

`train.py` chama o `prepare.py` sozinho. Use `--no-prepare` para reaproveitar um
dataset já montado, ou `--prepare-only` para montar sem treinar.

### Cenários e volume de dados

O cenário de N minutos usa as pastas `min1 .. minN` — cumulativo, como a seção
4.4 do TCC exige (o conjunto de 10 min contém o de 5 min).

Antes de montar qualquer coisa, o `prepare.py` valida cada pasta contra o
`train.txt` do CVAT e aborta com a lista completa do que falta. Se um `.zip`
ainda estiver na pasta de origem, a mensagem avisa que a extração pode estar em
andamento. `--allow-partial` segue com o que existir.

### Split treino/validação

70/30 **por bloco temporal dentro de cada minuto**: os primeiros 70% dos quadros
vão para treino, os últimos 30% para validação (1260/540 no `ang`, 2520/1080 no
`broadcast`).

Quadros vizinhos são praticamente idênticos — um shuffle aleatório colocaria o
mesmo instante dos dois lados e inflaria o mAP. O corte temporal também é
estável entre cenários: o que é validação no conjunto de 5 min continua sendo
validação no de 40, o que torna as curvas comparáveis.

### Subamostragem (`--stride`)

O padrão **não é um número fixo**: é o stride que deixa a amostragem efetiva em
`TARGET_FPS` (10 fps) na condição escolhida — 3 no `ang` (30 fps), 6 no
`broadcast` (60 fps).

Isso é o que mantém a comparação honesta. Com um stride global de 3, o broadcast
entregaria 840 imagens de treino por minuto de vídeo contra 420 do ângulo
aberto: o dobro de amostras para o mesmo volume anotado, confundindo exatamente
a variável que o TCC mede. Com o padrão por condição, ambos dão **420 imagens de
treino por minuto de vídeo**.

O volume anotado continua sendo 5/10/20/40 minutos de vídeo — que é a dimensão
que o TCC mede —, muda só a densidade de amostragem.

Medido nesta máquina (GTX 1050 Ti, yolo11n, imgsz 640, batch 8). O gargalo muda
conforme o cenário cabe ou não no cache de RAM:

| cenário | treino / val   | throughput | 50 épocas @ stride 3 |
|---------|----------------|-----------|----------------------|
| 1 min   | 420 / 180      | ~24 img/s (GPU)   | ~0,3 h        |
| 5 min   | 2.100 / 900    | ~23 img/s (GPU)   | **2,5 h** (medido) |
| 10 min  | 4.200 / 1.800  | ~15 img/s?        | ~4–8 h        |
| 20 min  | 8.400 / 3.600  | ~10 img/s?        | ~12–18 h      |
| 40 min  | 16.800 / 7.200 | **~8 img/s** (disco) | **~29 h** (medido) |

Os dois valores em negrito são runs reais completos; os do meio são interpolação
entre eles. Com `--stride 1` multiplique por 3.

O cenário de 40 min é limitado pela **leitura do disco externo**, não pela GPU:
33 GB de PNG não cabem no cache do sistema, então cada época relê do disco. Se
o prazo apertar, mover `D:\tcc\angulo_aberto` para um SSD interno ajuda muito
mais do que qualquer ajuste de batch.

### Montagem por symlink

O dataset é montado em `datasets/<condição>_<NN>min_s<stride>/` só com symlinks
para o disco de origem — nada é copiado. Os nomes dos arquivos são preservados,
então dá para rastrear qualquer frame de volta ao minuto de origem.

No Windows isso exige o **Modo de Desenvolvedor** ligado (*Configurações >
Privacidade e segurança > Para desenvolvedores*). Sem ele, use
`--link-mode copy`.

Um `manifest.json` registra o que foi montado; rodar de novo com os mesmos
argumentos pula a remontagem (`--force` refaz, `--clean` apaga antes).

## Saídas

```
runs/detect/ang_05min_s3_default/
  weights/best.pt  last.pt  epoch010.pt  epoch025.pt  epoch050.pt
  results.csv  results.png  confusion_matrix.png  args.yaml
  run_meta.json          # condição, minutos, stride, aug, modelo, contagens
```

O nome do run é derivado dos argumentos, e `train.py` recusa sobrescrever um run
existente — para retomar, use `--resume`.

Para reler um checkpoint use `YOLO("runs/detect/.../weights/epoch025.pt")`;
`torch.load` direto falha no torch 2.5.1 (`weights_only=True` por padrão).

## Interrupções e retomada

Um treino de 12–24 h em notebook vai ser interrompido em algum momento. Duas
situações, com desfechos diferentes:

- **Hibernação / suspensão** — o processo é congelado e volta sozinho. Nada a
  fazer; a época em curso apenas fica com o tempo de relógio inflado.
- **Queda de energia ou kill** — o processo morre. Retome com:

```bash
python train.py 5 --resume
```

`--resume` continua da época seguinte à última concluída, mantendo o total
original de épocas, o estado do otimizador e o mesmo diretório de run. Ele
**herda as épocas de checkpoint do `run_meta.json`** do run anterior, então não
é preciso repetir `--checkpoints`. Se o run já tiver terminado, ele recusa com
uma mensagem explícita em vez de treinar de novo.

Validado de ponta a ponta: treino de 8 épocas morto à força na 4ª, retomado com
`--resume` puro → `results.csv` contínuo de 1 a 8 sem duplicatas, `box_loss`
caindo na fronteira (1,903 → 1,785, provando que os pesos e o otimizador foram
restaurados e não reiniciados), e os quatro snapshots gerados e reduzidos.

Duas armadilhas do Ultralytics que o `train.py` contorna:

- `resume=True` faz o Ultralytics chamar `get_latest_run()`, que dá glob em
  `./**/last*.pt` e escolhe **o mais recente por data** — com vários runs, ele
  retomaria o errado. O `train.py` passa o caminho explícito do `last.pt`.
- Ao retomar, o Ultralytics substitui os args dele pelos do checkpoint e só
  aceita sobrescrever uma lista curta (`imgsz`, `batch`, `device`, `workers`,
  `cache`, `patience`, …). `--epochs` e `--aug` na linha de comando são
  ignorados; por isso o `run_meta.json` grava os valores efetivos do trainer, e
  não os da CLI.

## Estrutura

| arquivo | papel |
|---------|-------|
| [`config.py`](config.py) | raízes de dados, geometria dos minutos, perfis de aumento, épocas de checkpoint |
| [`prepare.py`](prepare.py) | valida as pastas, faz o split e monta o dataset com symlinks |
| [`train.py`](train.py) | entrypoint: prepara, treina e salva os checkpoints |
