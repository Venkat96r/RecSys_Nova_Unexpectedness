"""
LightGCN – simplified graph convolution for collaborative filtering.
Used as the pre-training step to produce GNN-enriched item embeddings.

Reference: He et al., LightGCN (SIGIR 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int,
                 embed_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users  = num_users
        self.num_items  = num_items
        self.embed_dim  = embed_dim
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.norm_adj: torch.Tensor = None  # set by build_graph()

    # ─── Graph construction ──────────────────────────────────────────────────

    def build_graph(self, user_ids: torch.Tensor,
                    item_ids: torch.Tensor,
                    device: str):
        """Build symmetric normalised adjacency (stored as sparse COO tensor)."""
        N = self.num_users + self.num_items

        row = torch.cat([user_ids, item_ids + self.num_users]).to(device)
        col = torch.cat([item_ids + self.num_users, user_ids]).to(device)
        indices = torch.stack([row, col])

        # Degree normalisation: D^{-1/2} A D^{-1/2}
        deg = torch.zeros(N, device=device)
        deg.scatter_add_(0, row, torch.ones(len(row), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        self.norm_adj = torch.sparse_coo_tensor(
            indices.to(device), values.to(device),
            size=(N, N)
        ).coalesce()

    # ─── Forward (message passing) ───────────────────────────────────────────

    def forward(self):
        emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_emb = [emb]
        for _ in range(self.num_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            all_emb.append(emb)

        # Mean pooling across layers
        final = torch.stack(all_emb, dim=1).mean(dim=1)
        user_out = final[:self.num_users]
        item_out = final[self.num_users:]
        return user_out, item_out

    # ─── Convenience: get enriched item embeddings as numpy ─────────────────

    @torch.no_grad()
    def get_item_embeddings(self) -> np.ndarray:
        _, item_out = self.forward()
        return item_out.cpu().numpy()
