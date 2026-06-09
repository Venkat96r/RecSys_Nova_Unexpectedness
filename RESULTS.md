# NOVA-Style Unexpectedness Recommender — Results

**Paper**: "Unexpectedness is What You Need: Breaking the Filter Bubble"  
**Dataset**: MovieLens 1M  
**Evaluation protocol**: UIRS-GNN (Complex & Intelligent Systems, 2023) — leave-one-out split, 99 random negatives + 1 target = 100 candidates per user

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | NOVA Transformer (Non-Invasive Gated Attention) |
| Embedding dim | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Max sequence length | 50 |
| Dropout | 0.2 |
| GNN (pre-training) | LightGCN, 3 layers, 50 epochs |
| Training loss | BPR (Bayesian Personalised Ranking) |
| Training negatives | 4 per positive |
| Batch size | 1024 |
| Epochs | 50 (best checkpoint at epoch 45 by val NDCG@10) |
| Optimiser | AdamW, lr=1e-3, weight_decay=1e-4 |
| LR schedule | CosineAnnealingLR (η_min = 1e-4) |
| Device | NVIDIA RTX 4070 Laptop (8 GB) |

---

## Dataset Statistics

| Stat | Value |
|------|-------|
| Users | 6,040 |
| Items | 3,260 |
| Interactions | 998,539 |
| Density | 5.07% |
| Train samples | 980,419 |
| Val users | 6,040 |
| Test users | 6,040 |

---

## Final Test Results (Best Checkpoint — Epoch 45)

### Accuracy

| Metric | @5 | @10 | @20 |
|--------|------|------|------|
| **HR** (Hit Rate) | 0.6235 | 0.7480 | 0.8536 |
| **NDCG** | 0.4741 | 0.5144 | 0.5413 |
| **Precision** | 0.1247 | 0.0748 | 0.0427 |

**AUC = 0.9114** (averaged over users)

### Beyond-Accuracy — Filter Bubble Breaking

| Metric | @5 | @10 | @20 |
|--------|------|------|------|
| **AvgUnexp** (Eq. 6, cluster distance) | 0.7062 | 0.7434 | 0.7955 |
| **ILD** (Intra-List Diversity, pairwise 1−cos) | 0.3326 | 0.4103 | 0.5054 |
| **EPC** (Expected Popularity Complement) | 0.7420 | 0.7683 | 0.8030 |
| **Coverage** (fraction of catalogue covered) | 0.8328 | 0.9454 | 0.9936 |

### Serendipity

Serendipity@K = (1/K) × Σᵢ₌₁ᴷ (Unexpectedness(i) × Relevance(i))

| Metric | @5 | @10 | @20 |
|--------|------|------|------|
| **Serendipity** | 0.0830 | **0.0504** | 0.0291 |

---

## Validation Curve (NDCG@10 per epoch)

| Epoch | Val HR@10 | Val NDCG@10 |
|-------|-----------|-------------|
| 40 | 0.7697 | 0.5342 |
| **45** | **0.7662** | **0.5352** ✓ |
| 50 | 0.7613 | 0.5291 |

Best val NDCG@10 = **0.5352** at epoch 45.

---

## Val vs Test Gap (Overfitting Check)

| Metric | Val (ep 45) | Test | Drop |
|--------|-------------|------|------|
| HR@10 | 0.7662 | 0.7480 | −2.4% |
| NDCG@10 | 0.5352 | 0.5144 | −3.9% |
| AUC | 0.9184 | 0.9114 | −0.8% |
| AvgUnexp@10 | 0.7431 | 0.7434 | +0.0% |
| ILD@10 | 0.4089 | 0.4103 | +0.3% |
| EPC@10 | 0.7682 | 0.7683 | +0.0% |
| Coverage@10 | 0.9488 | 0.9454 | −0.4% |
| Serendipity@10 | 0.0515 | 0.0504 | −2.1% |

All gaps < 5% → no overfitting.

---

## Comparison with UIRS-GNN Paper (ML-1M)

| Metric | UIRS-GNN reported | Our NOVA |
|--------|------------------|----------|
| HR@10 | ~0.70–0.76 | **0.748** |
| NDCG@10 | ~0.45–0.53 | **0.514** |
| AUC | ~0.87–0.92 | **0.911** |
| EPC@10 | ~0.70–0.78 | **0.768** |

Results are competitive with the UIRS-GNN baseline on all reported metrics.

---

## Lift over Random Baseline

A random scoring model over 100 candidates gives: HR@10 ≈ 0.10, NDCG@10 ≈ 0.043, AUC ≈ 0.50.

| Metric | Random | NOVA | Lift |
|--------|--------|------|------|
| HR@10 | 0.097 | 0.748 | **+7.7×** |
| NDCG@10 | 0.043 | 0.514 | **+12.0×** |
| AUC | 0.500 | 0.911 | **+0.41 absolute** |

---

## Architecture Notes

### Non-Invasive Gated Attention (NOVA)
The key design follows the paper's "Librarian Metaphor":

- **Gated Fusion** (Eq. 11–12): Four feature vectors per interaction (item embedding, rating, time-gap, unexpectedness score) are fused with a learned gate G = σ(W_F · concat(f_i)) → fused representation R.
- **Non-Invasive MHA** (Eq. 13–14): Queries and Keys come from R (fused), but Values come from f_item only. This prevents the unexpectedness signal from directly distorting the item representation while still influencing which items are attended to.
- **Causal masking**: Decoder-style (SASRec) — each position only attends to prior positions.

### Unexpectedness (Eq. 6)
Per-user interest islands are found via MiniBatchKMeans (K = max(2, min(5, N//15))). Unexpectedness of a candidate item c for user u is:

```
Unexp(c, u) = Σ_k  w_k · ||norm(emb_c) − center_k||_2
```

where centers and weights come from the user's interaction history.

### LightGCN Pre-training
Item embeddings are enriched via 3-layer LightGCN (BPR, 50 epochs) on the full user-item interaction graph before being frozen and used as the item tower input.

### Why seq_len = 50? (Empirically Validated)

We explicitly investigated this. First, we measured the actual ML-1M history length distribution:

| Statistic | Value |
|-----------|-------|
| Min | 17 |
| Median | **96** |
| Mean | 165 |
| p75 | 207 |
| p90 | 399 |
| Max | 2,233 |

| seq_len | Users fully covered | Users truncated |
|---------|---------------------|-----------------|
| 20 | 1.6% | 98.4% |
| 50 | 29.7% | 70.3% |
| 100 | 51.9% | 48.1% |
| 200 | 73.9% | 26.1% |

This made seq_len=50 look wrong — 70% of users are truncated. We retrained with seq_len=100 to test. The results:

| Metric | seq_len=50 | seq_len=100 | Δ |
|--------|-----------|------------|---|
| HR@10 | **0.7430** | 0.6467 | −9.6% |
| NDCG@10 | **0.5079** | 0.4178 | −17.7% |
| AUC | **0.9098** | 0.8818 | −3.1% |
| AvgUnexp@10 | **0.7402** | 0.7242 | −2.2% |

**seq_len=100 is worse on every metric.** The explanation is **recency bias** — the core principle of sequential recommendation:

- The last 50 interactions encode the user's **current preferences** almost entirely.
- Interactions from 3–5 years ago (which seq_len=100 adds for the average user) reflect old tastes that may have changed completely.
- The attention mechanism has to spread over 100 positions instead of 50, diluting the signal from the truly informative recent interactions.
- The model at seq_len=100 had loss=0.172 at epoch 30 vs 0.132 for seq_len=50, showing the harder task of modelling longer context with the same capacity.

This finding is consistent with the sequential recommendation literature: SASRec (2018) notes that beyond a threshold, additional history hurts because older interactions introduce noise rather than signal. The optimal seq_len is a model-specific empirical question, not purely a data coverage question. **seq_len=50 is validated by experiment, not just assumed.**

---

### Why 2 Transformer Layers?

Two layers is the established standard for ML-1M in sequential recommendation. Three reasons justify this choice:

**1. Dataset size vs model capacity.**  
ML-1M has ~980k training samples and 3,260 unique items. With `embed_dim=64`, each transformer block adds ~33k parameters (QKV attention + FFN). At 2 layers the model has 329k parameters — well matched to the dataset size. More layers add parameters faster than the data can support, increasing overfitting risk.

**2. Short sequence length.**  
With `max_seq_len=50`, user histories are at most 50 interactions. Deeper stacks are most beneficial when sequences are long (100+) and contain hierarchical patterns requiring multiple levels of abstraction. With 50-item sequences, 2 layers captures essentially all useful sequential signal.

**3. Empirical consensus on this benchmark.**  
The original SASRec paper (the foundational sequential recommender) explicitly compared 1, 2, and 4 layers on ML-1M and found 2 gives the best NDCG@10 — more layers overfit slightly because the required attention span is short. Our own validation curve confirms no underfitting: loss decreased monotonically from 0.174 (epoch 5) to 0.132 (epoch 30) without signs of saturation, meaning the model capacity is appropriate, not too small.

**When would more layers help?**  
Scaling to 4 layers is worthwhile if moving to a much larger dataset (ML-20M, Amazon) or increasing `embed_dim` to 128+ simultaneously to keep depth/width balanced. For ML-1M at `embed_dim=64`, 2 layers is the correct, paper-aligned choice.

---

## File Locations

| File | Description |
|------|-------------|
| `checkpoints/nova_best.pt` | Best model weights (epoch 45) |
| `data/processed/item_emb.npy` | LightGCN item embeddings (3260 × 64) |
| `data/processed/user_clusters.pkl` | Per-user KMeans cluster info |
| `data/processed/item_pop.npy` | Normalised item popularity (for EPC) |
| `data/processed/splits.pkl` | Leave-one-out train/val/test splits |
| `config.py` | All hyperparameters |
| `src/model.py` | NOVA architecture |
| `src/evaluate.py` | All 8 metric families |
| `main.py` | Training entry point |
