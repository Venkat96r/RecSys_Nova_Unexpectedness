"""Quick smoke-test for the updated UIRS-GNN-aligned evaluation pipeline."""
import sys, numpy as np
sys.path.insert(0, '.')

from src.mean_shift import user_unexpectedness_and_clusters
from src.evaluate import _batch_unexp, evaluate, print_metrics
from src.dataset import SeqRecDataset
from src.model import NOVARecModel
from torch.utils.data import DataLoader

rng = np.random.default_rng(0)

# 1. Mean Shift returns clusters correctly
embs = rng.standard_normal((40, 32)).astype('f')
scores, centers, weights = user_unexpectedness_and_clusters(embs)
print(f'MeanShift OK  scores={scores.shape} centers={centers.shape} weights={weights.shape}')
assert scores.shape == (40,)
cand_unexp = _batch_unexp(embs[:5], centers, weights)
assert cand_unexp.shape == (5,), f'bad _batch_unexp shape {cand_unexp.shape}'
print(f'_batch_unexp OK  shape={cand_unexp.shape}')

# 2. Dataset contains uid
NUM_U, NUM_I = 30, 80
splits = {}
user_clusters = {}
item_pop = rng.random(NUM_I).astype('f')

for uid in range(NUM_U):
    seq = [(int(rng.integers(1, NUM_I)), float(rng.integers(1, 6)),
            1_000_000 + t * 3600, int(rng.integers(0, 32)), float(rng.random()))
           for t in range(20)]
    splits[uid] = {'train': seq[:17], 'val': seq[17:19], 'test': seq[19:]}
    embs_u = rng.standard_normal((17, 32)).astype('f')
    _, ctrs, wts = user_unexpectedness_and_clusters(embs_u)
    user_clusters[uid] = {'centers': ctrs, 'weights': wts}

val_ds = SeqRecDataset(splits, 'val', NUM_I, num_negatives=99, max_seq_len=20)
b0 = val_ds[0]
assert 'uid' in b0, 'uid missing from dataset batch'
print(f'Dataset uid key present, uid={int(b0["uid"])}')

# 3. evaluate() returns all five metric families
val_loader = DataLoader(val_ds, batch_size=8)
item_emb_np = rng.standard_normal((NUM_I, 32)).astype('f')
model = NOVARecModel(NUM_I, 32, 4, 1, 20, 32, 0.0, item_emb_np)

metrics = evaluate(model, val_loader, item_emb_np, item_pop,
                   user_clusters, [5, 10, 20], 'cpu')
print()
print_metrics(metrics, 'Validation')

expected = {'HR@5','HR@10','NDCG@5','NDCG@10',
            'AvgUnexp@5','AvgUnexp@10',
            'ILD@5','ILD@10',
            'EPC@5','EPC@10'}
missing = expected - set(metrics.keys())
assert not missing, f'Missing metric keys: {missing}'

print()
print('ALL UIRS-GNN METRIC TESTS PASSED')
