"""
Parâmetros de cada algoritmo de rastreamento, nas duas configurações:

  default   os valores da documentação/repositório oficial de cada um
  custom    a segunda configuração do TCC, tirada da literatura de esportes e de cenas com oclusão
            (detalhes e citações no doc "Configurações de SORT, DeepSORT e ByteTrack na literatura")

O run.py escolhe com --algorithm e --config. Ex.: --algorithm bytetrack --config custom
"""

TRACKER_CONFIGS = {
    "bytetrack": {
        # Padrão do Ultralytics (ultralytics/cfg/trackers/bytetrack.yaml)
        "default": {
            "track_high_thresh": 0.25,  # caixas acima disso entram na 1ª associação
            "track_low_thresh": 0.1,    # caixas entre este e o de cima só recuperam rastros que já existem
            "new_track_thresh": 0.25,   # confiança mínima para nascer um ID novo
            "track_buffer": 30,         # quantos frames um rastro perdido fica guardado esperando voltar
            "match_thresh": 0.8,        # limite para aceitar uma ligação caixa-rastro (maior = mais permissivo)
            "fuse_score": True,         # mistura a confiança da detecção com o IoU
        },
        # Literatura de esportes / oclusão
        "custom": {
            "track_high_thresh": 0.6,   # ByteTrack (Zhang et al., ECCV 2022, Seção 4.1); Deep-EIoU (WACVW 2024, 4.4);
                                        # DiffMOT (CVPR 2024, 4.1); SparseTrack (TCSVT 2025, IV.A)
            "track_low_thresh": 0.1,    # Deep-EIoU (4.4); código oficial do ByteTrack
            "new_track_thresh": 0.7,    # código oficial do ByteTrack e do BoT-SORT (limiar alto + 0.1)
            "track_buffer": 60,         # Deep-EIoU (4.4) e SparseTrack (IV.A): rastro perdido guardado por 60 frames
            "match_thresh": 0.8,        # ByteTrack (4.1, IoU >= 0.2); SoccerNet-Tracking (CVPRW 2022, Seção 4)
            "fuse_score": True,         # mantido como no default
        },
    },
    "deepsort": {
        # Padrão do repositório original (github.com/nwojke/deep_sort), pelo pacote deep-sort-realtime
        "default": {
            "max_iou_distance": 0.7,     # é distância (1 - IoU): 0,7 aceita IoU a partir de 0,3
            "max_age": 30,               # quantos frames um rastro sobrevive sem detecção
            "n_init": 3,                 # detecções seguidas para confirmar um rastro novo
            "max_cosine_distance": 0.2,  # quanto a aparência pode variar e ainda ser a mesma pessoa
            "nn_budget": 100,            # quantas "fotos" de aparência guarda de cada rastro
            "min_confidence": 0.3,       # confiança mínima da detecção (exemplo do README)
            "nms_max_overlap": 1.0,      # 1.0 = sem NMS extra
            "embedder": "mobilenet",     # rede de aparência padrão do deep-sort-realtime
        },
        # Literatura de esportes / oclusão
        "custom": {
            "max_iou_distance": 0.7,     # mantido como no default
            "max_age": 60,               # Deep-EIoU (WACVW 2024, 4.4); o DeepSORT original usa 30 (ICIP 2017, Seção 3)
            "n_init": 3,                 # DeepSORT (2.1): "tentative during their first three frames"; StrongSORT mantém
            "max_cosine_distance": 0.2,  # mantido: StrongSORT (TMM 2023) só sobe para 0.45 com re-ID forte + custo de movimento
            "nn_budget": 100,            # DeepSORT (2.2): galeria de 100 descritores
            "min_confidence": 0.6,       # StrongSORT (TMM 2023, Seção V.A)
            "nms_max_overlap": 1.0,      # mantido como no default
            "embedder": "mobilenet",     # mantido: a rede de aparência é a mesma nas duas configurações
        },
    },
    "sort": {
        # Padrão do repositório original (github.com/abewley/sort, classe Sort)
        "default": {
            "max_age": 1,            # quantos frames um rastro sobrevive sem detecção
            "min_hits": 3,           # detecções seguidas para confirmar um rastro novo
            "iou_threshold": 0.3,    # sobreposição mínima para ligar caixa e rastro
            "min_confidence": 0.25,  # o SORT não define; usei o conf padrão do Ultralytics
        },
        # Literatura de esportes / oclusão
        "custom": {
            "max_age": 30,           # OC-SORT (CVPR 2023, código); Huang et al. (WACVW 2023, 4.2.2); A_max do DeepSORT
            "min_hits": 3,           # código do SORT; equivale aos 3 frames "tentative" do DeepSORT (2.1)
            "iou_threshold": 0.3,    # OC-SORT (5.1); Huang et al. (4.2.2); Koeferl et al. (BMTT 2020, Seção 3)
            "min_confidence": 0.5,   # SORT original (ICIP 2016, Seção 3.1): detecções acima de 50%
        },
    },
}
