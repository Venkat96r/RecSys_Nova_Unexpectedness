"""
Full preprocessing pipeline for MovieLens 1M.

Outputs (saved to processed_dir):
  interactions.pkl  – DataFrame with columns:
                        uid, iid, rating, timestamp, time_gap_bucket, unexp_score
  user2idx.pkl      – dict raw_uid -> 0-based idx
  item2idx.pkl      – dict raw_iid -> 0-based idx
  item_genres.npy   – (num_items, num_genres) binary genre matrix
  user_sequences.pkl – dict uid -> list of (iid, rating, timestamp,
                                             time_gap_bucket, unexp_score)
  lightgcn_item_emb.npy – (num_items, embed_dim) pre-trained item embeddings
  splits.pkl        – dict with 'train'/'val'/'test' per uid
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from data.download import download_ml1m, download_ml20m


# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(data_dir: str):
    """Load ML-1M (.dat files, :: separator)."""
    ratings_path = os.path.join(data_dir, 'ratings.dat')
    movies_path  = os.path.join(data_dir, 'movies.dat')

    ratings = pd.read_csv(
        ratings_path, sep='::', engine='python', header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    movies = pd.read_csv(
        movies_path, sep='::', engine='python', header=None,
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    return ratings, movies


def load_raw_ml20m(data_dir: str):
    """Load ML-20M (.csv files, comma separator with header)."""
    ratings = pd.read_csv(
        os.path.join(data_dir, 'ratings.csv'),
        dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': int}
    ).rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})

    movies = pd.read_csv(
        os.path.join(data_dir, 'movies.csv')
    ).rename(columns={'movieId': 'movie_id'})

    return ratings, movies


# ─────────────────────────────────────────────────────────────────────────────
# 2. Filtering & ID remapping
# ─────────────────────────────────────────────────────────────────────────────

def filter_and_remap(ratings: pd.DataFrame, min_interactions: int = 10):
    # k-core filtering
    while True:
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['movie_id'].value_counts()
        ratings = ratings[
            ratings['user_id'].isin(user_counts[user_counts >= min_interactions].index) &
            ratings['movie_id'].isin(item_counts[item_counts >= min_interactions].index)
        ]
        new_uc = ratings['user_id'].value_counts()
        new_ic = ratings['movie_id'].value_counts()
        if (new_uc >= min_interactions).all() and (new_ic >= min_interactions).all():
            break

    users = sorted(ratings['user_id'].unique())
    items = sorted(ratings['movie_id'].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}

    ratings = ratings.copy()
    ratings['uid'] = ratings['user_id'].map(user2idx)
    ratings['iid'] = ratings['movie_id'].map(item2idx)
    ratings = ratings.sort_values(['uid', 'timestamp']).reset_index(drop=True)

    return ratings, user2idx, item2idx


# ─────────────────────────────────────────────────────────────────────────────
# 3. Genre matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_genre_matrix(movies: pd.DataFrame, item2idx: dict):
    num_items = len(item2idx)
    # Collect all genres
    all_genres = set()
    for g in movies['genres']:
        all_genres.update(g.split('|'))
    all_genres = sorted(all_genres)
    genre2id = {g: i for i, g in enumerate(all_genres)}

    genre_mat = np.zeros((num_items, len(all_genres)), dtype=np.float32)
    for _, row in movies.iterrows():
        if row['movie_id'] not in item2idx:
            continue
        idx = item2idx[row['movie_id']]
        for g in row['genres'].split('|'):
            if g in genre2id:
                genre_mat[idx, genre2id[g]] = 1.0

    return genre_mat, genre2id


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-gap discretization
# ─────────────────────────────────────────────────────────────────────────────

def discretize_time_gaps(timestamps: np.ndarray,
                         num_buckets: int = 256,
                         t_min: int = 60,
                         t_max: int = 3 * 365 * 24 * 3600) -> np.ndarray:
    """Log-spaced bucketing; bucket 0 = first interaction (no gap)."""
    gaps = np.diff(timestamps, prepend=timestamps[0])
    gaps[0] = 0
    gaps = np.clip(gaps, 0, None)

    log_min = np.log1p(t_min)
    log_max = np.log1p(t_max)
    log_gaps = np.log1p(gaps.astype(np.float64))
    bucket_ids = np.floor(
        (log_gaps - log_min) / (log_max - log_min) * (num_buckets - 2)
    ).astype(int)
    bucket_ids = np.clip(bucket_ids, 0, num_buckets - 2)
    bucket_ids[0] = num_buckets - 1  # special "first-interaction" bucket
    return bucket_ids.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pre-train LightGCN to get item embeddings
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_lightgcn(ratings: pd.DataFrame,
                      num_users: int,
                      num_items: int,
                      embed_dim: int = 64,
                      num_layers: int = 3,
                      epochs: int = 100,
                      lr: float = 1e-3,
                      device: str = 'cpu',
                      gnn_batch_size: int = 4096) -> np.ndarray:
    """
    BPR-trained LightGCN.  Returns item embedding matrix (num_items, embed_dim).

    GNN forward is called ONCE per epoch (not per mini-batch) so it scales to
    large graphs (ML-20M has 40M edges; per-batch forward would be too slow).
    """
    from src.lightgcn import LightGCN

    print('[pretrain] Building LightGCN interaction graph ...')
    user_ids = torch.tensor(ratings['uid'].values, dtype=torch.long)
    item_ids = torch.tensor(ratings['iid'].values, dtype=torch.long)

    model = LightGCN(num_users, num_items, embed_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.build_graph(user_ids, item_ids, device)

    pos_u = ratings['uid'].values.astype(np.int64)
    pos_i = ratings['iid'].values.astype(np.int64)
    n_pos = len(pos_u)
    rng = np.random.default_rng(42)

    print(f'[pretrain] Training LightGCN for {epochs} epochs '
          f'(batch={gnn_batch_size}, {n_pos:,} pairs) ...')

    for epoch in range(1, epochs + 1):
        # ── Graph propagation: ONCE per epoch ───────────────────────────────
        user_all, item_all = model()   # (nu, d), (ni, d) — full GNN forward

        perm = rng.permutation(n_pos)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n_pos, gnn_batch_size):
            idx = perm[start:start + gnn_batch_size]
            u  = torch.tensor(pos_u[idx], dtype=torch.long, device=device)
            pi = torch.tensor(pos_i[idx], dtype=torch.long, device=device)
            ni = torch.tensor(rng.integers(0, num_items, len(idx)),
                              dtype=torch.long, device=device)

            eu = user_all[u]
            ep = item_all[pi]
            en = item_all[ni]

            pos_score = (eu * ep).sum(dim=1)
            neg_score = (eu * en).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            loss += 1e-4 * (eu.norm(2).pow(2) + ep.norm(2).pow(2)
                            + en.norm(2).pow(2)) / len(idx)

            optimizer.zero_grad()
            loss.backward(retain_graph=(start + gnn_batch_size < n_pos))
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f'  epoch {epoch:3d}/{epochs}  '
                  f'BPR loss = {total_loss / n_batches:.4f}')

    with torch.no_grad():
        _, item_emb = model()
    return item_emb.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-user Mean Shift → offline unexpectedness scores
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_user(uid, seq, item_emb, bandwidth, min_samples):
    """Worker for parallel unexpectedness computation (must be top-level for pickling)."""
    from src.mean_shift import user_unexpectedness_and_clusters
    if len(seq) == 0:
        d = item_emb.shape[1]
        return uid, np.array([], dtype=np.float32), \
               np.zeros((1, d), dtype=np.float32), np.ones(1, dtype=np.float32)
    iids = np.array([s if isinstance(s, (int, np.integer)) else s[0]
                     for s in seq], dtype=np.int64)
    embs = item_emb[iids]
    scores, centers, weights = user_unexpectedness_and_clusters(
        embs, bandwidth=bandwidth, min_samples=min_samples)
    return uid, scores, centers, weights


def compute_unexpectedness(user_sequences: dict,
                           item_emb: np.ndarray,
                           bandwidth=None,
                           min_samples: int = 3):
    """
    Parallel per-user KMeans clustering -> unexpectedness scores + cluster info.

    Returns
    -------
    unexp_dict    : uid -> (T,) float32 per-interaction unexpectedness scores
    cluster_dict  : uid -> {'centers': (K,d), 'weights': (K,)}
    """
    from joblib import Parallel, delayed
    print('[unexp] Computing per-user unexpectedness (parallel KMeans) ...')

    items = list(user_sequences.items())
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_process_one_user)(uid, seq, item_emb, bandwidth, min_samples)
        for uid, seq in tqdm(items, desc='KMeans clusters')
    )

    unexp_dict, cluster_dict = {}, {}
    for uid, scores, centers, weights in results:
        unexp_dict[uid]   = scores
        cluster_dict[uid] = {'centers': centers, 'weights': weights}
    return unexp_dict, cluster_dict


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sequence assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(ratings: pd.DataFrame,
                    unexp_dict: dict,
                    num_time_buckets: int = 256,
                    t_min: int = 60,
                    t_max: int = 3 * 365 * 24 * 3600) -> dict:
    """
    Returns dict uid -> list of tuples:
        (iid, rating, timestamp, time_gap_bucket, unexp_score)
    """
    sequences = {}
    for uid, group in ratings.groupby('uid'):
        group = group.sort_values('timestamp')
        iids      = group['iid'].values
        rts       = group['rating'].values.astype(np.float32)
        ts        = group['timestamp'].values
        tgaps     = discretize_time_gaps(ts, num_time_buckets, t_min, t_max)
        unexp     = unexp_dict.get(uid, np.zeros(len(iids), dtype=np.float32))

        seq = list(zip(iids.tolist(),
                       rts.tolist(),
                       ts.tolist(),
                       tgaps.tolist(),
                       unexp.tolist()))
        sequences[uid] = seq
    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# 8. Train / val / test split  (leave-last-two-out)
# ─────────────────────────────────────────────────────────────────────────────

def split_sequences(sequences: dict) -> dict:
    splits = {}
    for uid, seq in sequences.items():
        if len(seq) < 3:
            continue
        splits[uid] = {
            'train': seq[:-2],
            'val':   seq[-2:-1],
            'test':  seq[-1:]
        }
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(cfg):
    import os, pickle
    os.makedirs(cfg.processed_dir, exist_ok=True)

    # Check if already done
    done_flag = os.path.join(cfg.processed_dir, 'done.flag')
    if os.path.isfile(done_flag):
        print('[preprocess] Processed data found – loading from cache.')
        return _load_processed(cfg.processed_dir)

    # 1. Download
    dataset = getattr(cfg, 'dataset', 'ml-1m')
    parent  = os.path.dirname(cfg.data_dir)
    if dataset == 'ml-20m':
        data_dir = download_ml20m(parent)
    else:
        data_dir = download_ml1m(parent)

    # 2. Load raw
    print('[preprocess] Loading raw data ...')
    if dataset == 'ml-20m':
        ratings, movies = load_raw_ml20m(data_dir)
    else:
        ratings, movies = load_raw(data_dir)

    # 3. Filter & remap
    print('[preprocess] Filtering and remapping IDs ...')
    ratings, user2idx, item2idx = filter_and_remap(ratings, cfg.min_interactions)
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f'  users={num_users}, items={num_items}, '
          f'interactions={len(ratings)}')

    # 4. Genre matrix
    genre_mat, _ = build_genre_matrix(movies, item2idx)

    # 5. Pre-train LightGCN
    item_emb = pretrain_lightgcn(
        ratings, num_users, num_items,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.gnn_layers,
        epochs=cfg.gnn_epochs,
        lr=cfg.gnn_lr,
        device=cfg.device,
        gnn_batch_size=getattr(cfg, 'gnn_batch_size', 4096),
    )

    # 6. Build preliminary sequences (without unexp yet) for Mean Shift
    prelim_seqs = {}
    for uid, group in ratings.groupby('uid'):
        group = group.sort_values('timestamp')
        prelim_seqs[uid] = group['iid'].values.tolist()

    # 7. Mean Shift unexpectedness + cluster info
    unexp_dict, cluster_dict = compute_unexpectedness(
        prelim_seqs,
        item_emb,
        bandwidth=cfg.ms_bandwidth,
        min_samples=cfg.ms_min_samples
    )

    # 7b. Item popularity (interaction count normalised to [0,1])
    item_pop = ratings['iid'].value_counts().reindex(
        range(num_items), fill_value=0).values.astype(np.float32)
    item_pop = item_pop / item_pop.max() if item_pop.max() > 0 else item_pop

    # 8. Full sequences
    sequences = build_sequences(
        ratings, unexp_dict,
        cfg.num_time_buckets, cfg.time_min_sec, cfg.time_max_sec
    )

    # 9. Splits
    splits = split_sequences(sequences)

    # 10. Save
    print('[preprocess] Saving processed data ...')
    with open(os.path.join(cfg.processed_dir, 'user2idx.pkl'), 'wb') as f:
        pickle.dump(user2idx, f)
    with open(os.path.join(cfg.processed_dir, 'item2idx.pkl'), 'wb') as f:
        pickle.dump(item2idx, f)
    with open(os.path.join(cfg.processed_dir, 'sequences.pkl'), 'wb') as f:
        pickle.dump(sequences, f)
    with open(os.path.join(cfg.processed_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
    with open(os.path.join(cfg.processed_dir, 'user_clusters.pkl'), 'wb') as f:
        pickle.dump(cluster_dict, f)
    np.save(os.path.join(cfg.processed_dir, 'item_emb.npy'), item_emb)
    np.save(os.path.join(cfg.processed_dir, 'genre_mat.npy'), genre_mat)
    np.save(os.path.join(cfg.processed_dir, 'item_pop.npy'), item_pop)

    open(done_flag, 'w').close()
    print('[preprocess] Done.')

    return {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'sequences': sequences,
        'splits': splits,
        'item_emb': item_emb,
        'genre_mat': genre_mat,
        'item_pop': item_pop,
        'user_clusters': cluster_dict,
        'num_users': num_users,
        'num_items': num_items,
    }


def _load_processed(processed_dir: str) -> dict:
    import pickle
    # If old cache is missing new files, force reprocessing
    for fname in ('user_clusters.pkl', 'item_pop.npy'):
        if not os.path.isfile(os.path.join(processed_dir, fname)):
            os.remove(os.path.join(processed_dir, 'done.flag'))
            raise FileNotFoundError(
                f'[preprocess] Cache missing {fname} – deleting done.flag, '
                'please re-run to regenerate with updated outputs.')

    with open(os.path.join(processed_dir, 'user2idx.pkl'), 'rb') as f:
        user2idx = pickle.load(f)
    with open(os.path.join(processed_dir, 'item2idx.pkl'), 'rb') as f:
        item2idx = pickle.load(f)
    with open(os.path.join(processed_dir, 'sequences.pkl'), 'rb') as f:
        sequences = pickle.load(f)
    with open(os.path.join(processed_dir, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    with open(os.path.join(processed_dir, 'user_clusters.pkl'), 'rb') as f:
        user_clusters = pickle.load(f)
    item_emb  = np.load(os.path.join(processed_dir, 'item_emb.npy'))
    genre_mat = np.load(os.path.join(processed_dir, 'genre_mat.npy'))
    item_pop  = np.load(os.path.join(processed_dir, 'item_pop.npy'))
    return {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'sequences': sequences,
        'splits': splits,
        'item_emb': item_emb,
        'genre_mat': genre_mat,
        'item_pop': item_pop,
        'user_clusters': user_clusters,
        'num_users': len(user2idx),
        'num_items': len(item2idx),
    }
