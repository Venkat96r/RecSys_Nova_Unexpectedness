"""
Evaluation metrics aligned with UIRS-GNN (Complex & Intelligent Systems, 2023).

Protocol
--------
- Leave-one-out split (last item = test, second-to-last = val)
- 99 randomly sampled negatives per target item  → 100 total candidates
- All metrics averaged over users

Accuracy metrics (matching UIRS-GNN exactly)
--------------------------------------------
  HR@K    – Hit Rate:  1 if target appears in top-K, else 0
  NDCG@K  – Normalised Discounted Cumulative Gain: 1/log2(rank+1) if hit

Unexpectedness / Diversity metrics (matching UIRS-GNN exactly)
---------------------------------------------------------------
  AvgUnexp@K – Average unexpectedness score (Eq. 6: Mean Shift cluster distance)
               of the K items actually recommended (top-K by score).
               This is what UIRS-GNN reports as the "unexpectedness" of the list.

  ILD@K      – Intra-List Diversity: average pairwise (1 - cosine_sim)
               among the K recommended items.  Measures whether the list
               covers diverse interest regions.

  EPC@K      – Expected Popularity Complement: average (1 - normalised_popularity)
               of top-K items.  1.0 = all niche items; 0.0 = all blockbusters.
               UIRS-GNN uses this to show the model avoids popularity bias.

  Serendipity@K – (1/K) × Σᵢ₌₁ᴷ (Unexpectedness(i) × Relevance(i))
               i iterates over the K recommended items (rank 1..K).
               Unexpectedness(i): Eq. 6 cluster-distance score (how surprising item i is
                 relative to the user's historical interest islands).
               Relevance(i): indicator = 1 if user interacted with item i, 0 otherwise.
               In leave-one-out evaluation only the target item has Relevance=1, so:
                 Serendipity@K = hit_at_K × unexp(target) / K  (per user, then averaged).

What we deliberately do NOT use
---------------------------------
- Distance from sequence *mean* as unexpectedness: this is the "Giant Blob"
  problem explicitly criticised in the proposal.  The Mean Shift cluster-based
  distance (Eq. 6) is the correct, paper-aligned definition.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Unexpectedness helper: Eq. 6 applied to a batch of candidate embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _batch_unexp(cand_embs: np.ndarray,
                 centers: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """
    cand_embs : (C, d)  candidate item embeddings (raw, will be L2-normalised)
    centers   : (K, d)  user's cluster centres    (already L2-normalised)
    weights   : (K,)    normalised cluster weights

    Returns (C,) float32 unexpectedness scores via Eq. 6.
    """
    norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    cand_norm = cand_embs / norms                  # (C, d)

    # scores[c] = Σ_k weights[k] * ||cand_norm[c] - centers[k]||
    scores = np.zeros(len(cand_embs), dtype=np.float32)
    for k in range(len(centers)):
        diff = cand_norm - centers[k]              # (C, d)
        scores += np.linalg.norm(diff, axis=1).astype(np.float32) * weights[k]
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# ILD helper
# ─────────────────────────────────────────────────────────────────────────────

def _ild(item_embs: np.ndarray) -> float:
    """
    item_embs : (K, d)  embeddings of recommended items
    Returns average pairwise (1 - cosine_similarity).
    """
    K = len(item_embs)
    if K < 2:
        return 0.0
    norms = np.linalg.norm(item_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = item_embs / norms
    sim_mat = normed @ normed.T                    # (K, K)
    # upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(K, k=1)
    return float(np.mean(1.0 - sim_mat[triu_idx]))


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model,
             loader: DataLoader,
             item_emb: np.ndarray,           # (num_items, d) 0-indexed
             item_pop: np.ndarray,            # (num_items,)  normalised popularity
             user_clusters: dict,             # uid -> {'centers': (K,d), 'weights': (K,)}
             topk_list=(5, 10, 20),
             device: str = 'cpu') -> dict:
    """
    Evaluate the model using UIRS-GNN-aligned metrics.

    Parameters
    ----------
    model         : NOVARecModel
    loader        : DataLoader (batches must include 'uid')
    item_emb      : pre-trained item embedding matrix, 0-indexed
                    (index i corresponds to dataset item-id i+1)
    item_pop      : normalised popularity per item (0-indexed, same as item_emb)
    user_clusters : per-user Mean Shift cluster info from preprocessing
    topk_list     : list of K values to evaluate
    device        : 'cpu' or 'cuda'
    """
    model.eval()
    max_k     = max(topk_list)
    num_items = item_emb.shape[0]          # total catalogue size (0-indexed)

    per_user_keys = (
        [f'HR@{k}'            for k in topk_list] +
        [f'NDCG@{k}'          for k in topk_list] +
        ['AUC']               +                    # single global AUC
        [f'Precision@{k}'     for k in topk_list] +
        [f'AvgUnexp@{k}'      for k in topk_list] +
        [f'ILD@{k}'           for k in topk_list] +
        [f'EPC@{k}'           for k in topk_list] +
        [f'Serendipity@{k}'   for k in topk_list]
    )
    results = {k: [] for k in per_user_keys}

    # Coverage needs a global set per K (not per-user average)
    coverage_sets = {k: set() for k in topk_list}

    for batch in tqdm(loader, desc='Eval', leave=False):
        uids       = batch['uid'].numpy()
        seq_iid    = batch['seq_iid'].to(device)
        seq_rat    = batch['seq_rat'].to(device)
        seq_tgap   = batch['seq_tgap'].to(device)
        seq_unexp  = batch['seq_unexp'].to(device)
        pad_mask   = batch['pad_mask'].to(device)
        target_iid = batch['target_iid'].to(device)
        neg_iids   = batch['neg_iids'].to(device)

        hidden = model(seq_iid, seq_rat, seq_tgap, seq_unexp, pad_mask)
        h_last = model.gather_last(hidden, pad_mask)

        cand_iids = torch.cat([target_iid.unsqueeze(1), neg_iids], dim=1)  # (B,100)
        scores    = model.score_candidates(h_last, cand_iids)              # (B,100)

        B, C = scores.shape
        n_neg = C - 1                          # 99 negatives

        # ── Rank of the positive item (1-indexed) ────────────────────────────
        target_score = scores[:, 0:1]
        rank = (scores >= target_score).sum(dim=1).float().cpu().numpy()   # (B,)

        # ── HR, NDCG, AUC, Precision ────────────────────────────────────────
        # AUC = P(pos > neg) = 1 − (rank−1) / n_neg
        auc_vals = 1.0 - (rank - 1.0) / n_neg
        results['AUC'].extend(auc_vals.tolist())

        for k in topk_list:
            hit  = (rank <= k).astype(np.float32)
            ndcg = np.where(rank <= k, 1.0 / np.log2(rank + 1), 0.0).astype(np.float32)
            results[f'HR@{k}'].extend(hit.tolist())
            results[f'NDCG@{k}'].extend(ndcg.tolist())
            results[f'Precision@{k}'].extend((hit / k).tolist())

        # ── Top-K indices for beyond-accuracy metrics ────────────────────────
        effective_k = min(max_k, C)
        _, top_idx  = scores.topk(effective_k, dim=1)
        top_idx_np  = top_idx.cpu().numpy()
        cand_np     = cand_iids.cpu().numpy()

        for b in range(B):
            uid = int(uids[b])
            cand_item_ids_0 = cand_np[b] - 1          # (100,) 0-indexed
            cand_embs_np    = item_emb[cand_item_ids_0]

            clust = user_clusters.get(uid, None)
            if clust is not None:
                centers, weights = clust['centers'], clust['weights']
            else:
                centers = np.zeros((1, item_emb.shape[1]), dtype=np.float32)
                weights = np.ones(1, dtype=np.float32)

            unexp_all = _batch_unexp(cand_embs_np, centers, weights)
            # unexp_all[0] is always the target item (candidates = [target, neg×99])
            unexp_target = float(unexp_all[0])

            for k in topk_list:
                actual_k          = min(k, effective_k)
                top_k_local       = top_idx_np[b, :actual_k]
                top_k_item_0      = cand_item_ids_0[top_k_local]
                top_k_embs        = item_emb[top_k_item_0]

                results[f'AvgUnexp@{k}'].append(float(unexp_all[top_k_local].mean()))
                results[f'ILD@{k}'].append(_ild(top_k_embs))

                pop_vals = item_pop[top_k_item_0]
                results[f'EPC@{k}'].append(float(np.mean(1.0 - pop_vals)))

                # Coverage: accumulate recommended 0-indexed item ids globally
                coverage_sets[k].update(top_k_item_0.tolist())

                # Serendipity@K = (1/K) Σᵢ Unexpectedness(i)×Relevance(i)
                # Only the target has Relevance=1 → simplifies to hit×unexp_target/K
                hit_k = float(rank[b] <= k)
                results[f'Serendipity@{k}'].append(hit_k * unexp_target / k)

    out = {k: float(np.mean(v)) for k, v in results.items()}
    for k in topk_list:
        out[f'Coverage@{k}'] = len(coverage_sets[k]) / num_items
    return out


def print_metrics(metrics: dict, prefix: str = ''):
    groups = {
        'Accuracy':       ['HR', 'NDCG'],
        'Ranking':        ['AUC', 'Precision'],
        'Unexpectedness': ['AvgUnexp', 'ILD', 'EPC'],
        'Serendipity':    ['Serendipity'],
        'Coverage':       ['Coverage'],
    }
    line = prefix + '\n'
    for group_name, prefixes in groups.items():
        entries = [(k, v) for k, v in sorted(metrics.items())
                   if any(k.startswith(p) for p in prefixes)]
        if entries:
            line += f'  [{group_name}]'
            for k, v in entries:
                line += f'  {k}={v:.4f}'
            line += '\n'
    print(line.rstrip())
