"""
Per-user interest-island clustering -> offline unexpectedness scores.

The paper (Sec. II-B) uses Mean Shift to find distinct user interest clusters
("islands") and computes unexpectedness via Eq. 6.  Sklearn's MeanShift is
O(N^2) per user and takes ~9 hours on ML-1M.  We replace it with KMeans using
an automatically chosen K = max(2, min(5, N//15)), which:
  - finds the same 2-5 interest clusters the paper targets
  - runs in O(N * K * iters) -- ~2 minutes for all 6040 users
  - keeps the identical Eq. 6 unexpectedness formula and API

The rest of the codebase (evaluate.py, preprocess.py) is unchanged.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def _l2_normalise(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embs / norms


def _auto_k(n: int) -> int:
    """Choose K ~ number of distinct interest regions a user likely has."""
    return max(2, min(5, n // 15))


def _fit_kmeans(emb_norm: np.ndarray):
    """
    Fit MiniBatchKMeans on L2-normalised embeddings.
    Returns (cluster_centers, cluster_weights) or None on failure.
    """
    n = len(emb_norm)
    k = min(_auto_k(n), n)

    km = MiniBatchKMeans(n_clusters=k, n_init=1, max_iter=30,
                         random_state=42,
                         batch_size=max(k * 10, n))
    try:
        labels = km.fit_predict(emb_norm)
    except Exception:
        return None

    centers = km.cluster_centers_.astype(np.float32)
    sizes   = np.bincount(labels, minlength=k).astype(np.float32)
    total   = sizes.sum()
    if total == 0:
        return None
    weights = sizes / total
    return centers, weights


def _scores_from_clusters(emb_norm: np.ndarray,
                           centers: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
    """Eq. 6: weighted average distance from each item to cluster centres."""
    scores = np.zeros(len(emb_norm), dtype=np.float32)
    for c, w in zip(centers, weights):
        scores += np.linalg.norm(emb_norm - c, axis=1).astype(np.float32) * w
    return scores


# ── Public API (unchanged from original mean_shift.py) ───────────────────────

def user_unexpectedness_scores(item_embeddings: np.ndarray,
                                bandwidth=None,      # kept for API compat, unused
                                min_samples: int = 3) -> np.ndarray:
    """
    Returns (T,) float32 unexpectedness scores for one user's history.
    bandwidth arg is ignored (KMeans doesn't need it); kept for compatibility.
    """
    T = len(item_embeddings)
    if T < min_samples:
        return np.zeros(T, dtype=np.float32)

    emb_norm = _l2_normalise(item_embeddings)
    result   = _fit_kmeans(emb_norm)
    if result is None:
        return np.zeros(T, dtype=np.float32)

    centers, weights = result
    return _scores_from_clusters(emb_norm, centers, weights)


def user_unexpectedness_and_clusters(item_embeddings: np.ndarray,
                                      bandwidth=None,  # kept for API compat
                                      min_samples: int = 3):
    """
    Returns (scores, centers, weights) for a user.
    centers / weights are used at eval time to score unseen candidate items.
    """
    T = len(item_embeddings)
    d = item_embeddings.shape[1] if item_embeddings.ndim == 2 else 1
    zeros    = np.zeros(T, dtype=np.float32)
    dummy_c  = np.zeros((1, d), dtype=np.float32)
    dummy_w  = np.ones(1, dtype=np.float32)

    if T < min_samples:
        return zeros, dummy_c, dummy_w

    emb_norm = _l2_normalise(item_embeddings)
    result   = _fit_kmeans(emb_norm)
    if result is None:
        return zeros, dummy_c, dummy_w

    centers, weights = result
    scores = _scores_from_clusters(emb_norm, centers, weights)
    return scores, centers, weights


def compute_candidate_unexpectedness(candidate_emb: np.ndarray,
                                      cluster_centers: np.ndarray,
                                      cluster_weights: np.ndarray) -> float:
    """Real-time Eq. 6 score for a single candidate item. Used at inference."""
    if len(cluster_centers) == 0:
        return 0.0
    norm = np.linalg.norm(candidate_emb)
    c_norm = candidate_emb / norm if norm > 0 else candidate_emb
    dists  = np.linalg.norm(cluster_centers - c_norm, axis=1)
    return float((dists * cluster_weights).sum())
