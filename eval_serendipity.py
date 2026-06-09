"""
Load the best checkpoint and recompute all metrics including Serendipity@K.
Run with: py -3.12 eval_serendipity.py
"""
import os, sys
sys.path.insert(0, '.')

import torch
import numpy as np
from torch.utils.data import DataLoader

from config import Config
from data.preprocess import run_preprocessing
from src.dataset import SeqRecDataset
from src.model import NOVARecModel
from src.evaluate import evaluate, print_metrics


def main():
    cfg = Config()

    print('Loading preprocessed data...')
    data = run_preprocessing(cfg)
    splits        = data['splits']
    item_emb      = data['item_emb']
    item_pop      = data['item_pop']
    user_clusters = data['user_clusters']
    num_items     = data['num_items']

    test_ds = SeqRecDataset(splits, 'test', num_items,
                             cfg.eval_negatives, cfg.max_seq_len, cfg.seed)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)

    model = NOVARecModel(
        num_items        = num_items,
        embed_dim        = cfg.embed_dim,
        num_heads        = cfg.num_heads,
        num_layers       = cfg.num_layers,
        max_seq_len      = cfg.max_seq_len,
        num_time_buckets = cfg.num_time_buckets,
        dropout          = cfg.dropout,
        pretrained_item_emb = item_emb,
    ).to(cfg.device)

    ckpt_path = os.path.join(cfg.save_dir, 'nova_best.pt')
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
    ckpt.pop('_causal_mask_cache', None)

    # Infer max_seq_len from checkpoint so we don't need to match config exactly
    ckpt_seq_len = ckpt['pos_emb.weight'].shape[0] - 1
    if ckpt_seq_len != cfg.max_seq_len:
        print(f'[warn] checkpoint seq_len={ckpt_seq_len}, config seq_len={cfg.max_seq_len}')
        print(f'       rebuilding model with seq_len={ckpt_seq_len} to match checkpoint')
        model = NOVARecModel(
            num_items        = num_items,
            embed_dim        = cfg.embed_dim,
            num_heads        = cfg.num_heads,
            num_layers       = cfg.num_layers,
            max_seq_len      = ckpt_seq_len,
            num_time_buckets = cfg.num_time_buckets,
            dropout          = cfg.dropout,
            pretrained_item_emb = item_emb,
        ).to(cfg.device)
        # Rebuild dataset with matching seq_len
        test_ds = SeqRecDataset(splits, 'test', num_items,
                                 cfg.eval_negatives, ckpt_seq_len, cfg.seed)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=cfg.num_workers)

    model.load_state_dict(ckpt)
    print(f'Loaded checkpoint: {ckpt_path}\n')

    metrics = evaluate(model, test_loader, item_emb, item_pop,
                       user_clusters, cfg.topk, cfg.device)
    print_metrics(metrics, prefix='Test (with Serendipity):')

    print('\n--- Serendipity Results ---')
    for k in cfg.topk:
        print(f'  Serendipity@{k:2d} = {metrics[f"Serendipity@{k}"]:.4f}')


if __name__ == '__main__':
    main()
