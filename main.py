"""
Entry point: preprocess → train → evaluate the NOVA-style recommender.

Usage:
    python main.py
    python main.py --epochs 30 --embed_dim 64 --device cuda
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.preprocess import run_preprocessing
from src.dataset import SeqRecDataset
from src.model import NOVARecModel
from src.evaluate import evaluate, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Train', leave=False):
        seq_iid    = batch['seq_iid'].to(device)
        seq_rat    = batch['seq_rat'].to(device)
        seq_tgap   = batch['seq_tgap'].to(device)
        seq_unexp  = batch['seq_unexp'].to(device)
        pad_mask   = batch['pad_mask'].to(device)
        target_iid = batch['target_iid'].to(device)
        neg_iids   = batch['neg_iids'].to(device)

        hidden = model(seq_iid, seq_rat, seq_tgap, seq_unexp, pad_mask)
        h_last = model.gather_last(hidden, pad_mask)  # (B, d)

        # Positive score
        pos_emb = model.item_emb(target_iid)             # (B, d)
        pos_score = (h_last * pos_emb).sum(dim=1)         # (B,)

        # Negative scores (mean over sampled negatives)
        neg_emb   = model.item_emb(neg_iids)              # (B, N, d)
        neg_score = torch.bmm(neg_emb,
                              h_last.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # BPR loss: -log σ(pos - neg) averaged over negatives
        pos_expand = pos_score.unsqueeze(1).expand_as(neg_score)
        loss = -torch.log(torch.sigmoid(pos_expand - neg_score) + 1e-8).mean()

        # L2 regularisation on embeddings
        l2 = (pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2) +
              h_last.norm(2).pow(2)) / seq_iid.size(0)
        loss = loss + 1e-5 * l2

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ── 1. Preprocessing ─────────────────────────────────────────────────────
    print('=' * 60)
    print('NOVA-Style Unexpectedness Recommender  (MovieLens 1M)')
    print('=' * 60)

    data = run_preprocessing(cfg)
    splits        = data['splits']
    item_emb      = data['item_emb']       # (num_items, d) numpy
    item_pop      = data['item_pop']       # (num_items,) normalised popularity
    user_clusters = data['user_clusters']  # uid -> {'centers', 'weights'}
    num_items     = data['num_items']

    print(f'\nDataset: {num_items} items, '
          f'{len(splits)} users, device={cfg.device}\n')

    # ── 2. Datasets & loaders ────────────────────────────────────────────────
    train_ds = SeqRecDataset(splits, 'train', num_items,
                              cfg.num_negatives, cfg.max_seq_len, cfg.seed)
    val_ds   = SeqRecDataset(splits, 'val',   num_items,
                              cfg.eval_negatives, cfg.max_seq_len, cfg.seed)
    test_ds  = SeqRecDataset(splits, 'test',  num_items,
                              cfg.eval_negatives, cfg.max_seq_len, cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                               shuffle=True,  num_workers=cfg.num_workers,
                               pin_memory=(cfg.device == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                               shuffle=False, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                               shuffle=False, num_workers=cfg.num_workers)

    print(f'Train samples: {len(train_ds):,}  '
          f'Val: {len(val_ds):,}  Test: {len(test_ds):,}')

    # ── 3. Model ─────────────────────────────────────────────────────────────
    # item_emb from LightGCN is 0-indexed (item 0 maps to embedding row 0)
    # but in the dataset items are 1-indexed (0 = PAD)
    # We pass the full embedding array; model assigns row i+1 → item i
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

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {param_count:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)

    # ── 4. Training ──────────────────────────────────────────────────────────
    best_ndcg10 = 0.0
    best_epoch  = 0
    best_path   = os.path.join(cfg.save_dir, 'nova_best.pt')

    print('\n--- Training ---')
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        scheduler.step()

        if epoch % 5 == 0 or epoch == cfg.epochs:
            val_metrics = evaluate(model, val_loader, item_emb, item_pop,
                                   user_clusters, cfg.topk, cfg.device)
            ndcg10 = val_metrics.get('NDCG@10', 0.0)
            print_metrics(val_metrics,
                          prefix=f'Epoch {epoch:3d}  loss={train_loss:.4f}  val:')

            if ndcg10 > best_ndcg10:
                best_ndcg10 = ndcg10
                best_epoch  = epoch
                torch.save(model.state_dict(), best_path)

    print(f'\nBest val NDCG@10 = {best_ndcg10:.4f} at epoch {best_epoch}')

    # ── 5. Test evaluation ───────────────────────────────────────────────────
    print('\n--- Test Evaluation ---')
    model.load_state_dict(torch.load(best_path, map_location=cfg.device))
    test_metrics = evaluate(model, test_loader, item_emb, item_pop,
                            user_clusters, cfg.topk, cfg.device)
    print_metrics(test_metrics, prefix='Test:')

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--embed_dim',  type=int,   default=None)
    parser.add_argument('--num_layers', type=int,   default=None)
    parser.add_argument('--num_heads',  type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--device',     type=str,   default=None)
    parser.add_argument('--gnn_epochs', type=int,   default=None)
    args = parser.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)

    main(cfg)
