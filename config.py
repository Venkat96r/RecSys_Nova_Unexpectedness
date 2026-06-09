import torch


class Config:
    # ── Dataset  ('ml-1m' or 'ml-20m') ───────────────────────────────────────
    dataset         = 'ml-1m'
    data_dir        = './data/ml-1m'
    processed_dir   = './data/processed'
    min_interactions = 10

    # ── Embedding / Model ─────────────────────────────────────────────────────
    embed_dim       = 64
    num_heads       = 4            # must divide embed_dim
    num_layers      = 2
    max_seq_len     = 50
    dropout         = 0.2

    # ── LightGCN (pre-training) ───────────────────────────────────────────────
    gnn_layers      = 3
    gnn_epochs      = 100
    gnn_lr          = 1e-3
    gnn_neg         = 1
    gnn_batch_size  = 4096

    # ── Time-gap discretization ───────────────────────────────────────────────
    num_time_buckets = 256
    time_min_sec    = 60
    time_max_sec    = 3 * 365 * 24 * 3600

    # ── Mean Shift ────────────────────────────────────────────────────────────
    ms_bandwidth    = None
    ms_min_samples  = 3

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size      = 1024
    lr              = 1e-3
    weight_decay    = 1e-4
    epochs          = 50
    num_negatives   = 4
    eval_negatives  = 99           # UIRS-GNN protocol: 99 neg + 1 target

    # ── Evaluation ────────────────────────────────────────────────────────────
    topk            = [5, 10, 20]

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed            = 42
    device          = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers     = 0
    save_dir        = './checkpoints'
