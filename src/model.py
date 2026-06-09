"""
Non-Invasive Gated Attention (NOVA-style) Recommender System.

Architecture (per paper Section IV):
  1. Four d-dimensional feature vectors per interaction:
       f_item  – GNN-enriched item embedding
       f_rating – explicit rating (1-5) embedding
       f_time  – log-bucketed time-gap embedding
       f_unexp – offline Mean Shift distance, projected to d dims

  2. Gated Fusion (Eq. 11-12):
       F_cat = concat(f_item, f_rating, f_time, f_unexp)  # (B,T,4d)
       G     = σ(F_cat @ W_F)                             # (B,T,4)
       R     = Σ_i G[:,:,i:i+1] * f_i                    # (B,T,d)

  3. Non-Invasive Multi-Head Attention (Eq. 13-14):
       Q = R @ W_Q   (from fused representation)
       K = R @ W_K   (from fused representation)
       V = f_item @ W_V  (PURE item only – "Librarian Metaphor")
       causal masked self-attention

  4. Stacked Transformer encoder blocks with residuals + LayerNorm.

  5. Final score: dot product of last hidden state with candidate item emb.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Gated Fusion Module
# ─────────────────────────────────────────────────────────────────────────────

class GatedFusor(nn.Module):
    """
    Fuses 4 feature vectors via a learned gate.

    G     = σ( [f0; f1; f2; f3] @ W_F )   shape (B,T,4)
    R     = Σ_i G_i * f_i                  shape (B,T,d)
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_F    = nn.Linear(4 * embed_dim, 4, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, f_item: torch.Tensor,
                f_rating: torch.Tensor,
                f_time:   torch.Tensor,
                f_unexp:  torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_item, f_rating, f_time, f_unexp: (B, T, d)
        Returns:
            R: (B, T, d)  fused context-aware representation
        """
        F_cat = torch.cat([f_item, f_rating, f_time, f_unexp], dim=-1)  # (B,T,4d)
        G = torch.sigmoid(self.W_F(F_cat))                               # (B,T,4)

        g0, g1, g2, g3 = G[..., 0:1], G[..., 1:2], G[..., 2:3], G[..., 3:4]
        R = g0 * f_item + g1 * f_rating + g2 * f_time + g3 * f_unexp   # (B,T,d)
        return self.norm(self.dropout(R))


# ─────────────────────────────────────────────────────────────────────────────
# Non-Invasive Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class NonInvasiveMHA(nn.Module):
    """
    Q, K  come from fused representation R  (context-aware)
    V     comes from pure item embedding    (semantically clean)

    This is the "Librarian Metaphor":
      Q/K = rich contextual search query
      V   = pure book content, uncontaminated by side features
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)  # V from f_item
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, R: torch.Tensor,
                f_item: torch.Tensor,
                causal_mask: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            R        : (B, T, d)  fused representation  → Q, K
            f_item   : (B, T, d)  pure item embedding   → V
            causal_mask : (T, T) additive mask (0 or -inf)
            pad_mask : (B, T) bool, True = real token
        Returns:
            out : (B, T, d)
        """
        B, T, d = R.shape
        h, hd = self.num_heads, self.head_dim

        def split_heads(x):
            return x.view(B, T, h, hd).transpose(1, 2)   # (B,h,T,hd)

        Q = split_heads(self.W_Q(R))
        K = split_heads(self.W_K(R))
        V = split_heads(self.W_V(f_item))                 # V from pure item

        scores = (Q @ K.transpose(-2, -1)) / self.scale   # (B,h,T,T)

        # Causal mask (additive, upper-triangular = -inf)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask: mask out padded positions in K
        if pad_mask is not None:
            # (B,1,1,T) – padded positions get -1e9
            key_mask = (~pad_mask).unsqueeze(1).unsqueeze(2).float() * -1e9
            scores = scores + key_mask

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block with Non-Invasive Attention
# ─────────────────────────────────────────────────────────────────────────────

class NOVATransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = NonInvasiveMHA(embed_dim, num_heads, dropout)
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.ffn    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, R: torch.Tensor,
                f_item: torch.Tensor,
                causal_mask: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual
        attn_out = self.attn(self.norm1(R), f_item, causal_mask, pad_mask)
        R = R + attn_out
        R = R + self.ffn(self.norm2(R))
        return R


# ─────────────────────────────────────────────────────────────────────────────
# Full NOVA-Style Recommendation Model
# ─────────────────────────────────────────────────────────────────────────────

class NOVARecModel(nn.Module):
    """
    Full Non-Invasive Gated Attention recommender.

    Parameters
    ----------
    num_items        : vocabulary size (+ 1 for PAD at index 0)
    embed_dim        : d
    num_heads        : h
    num_layers       : number of Transformer blocks
    max_seq_len      : maximum sequence length (for positional encodings)
    num_time_buckets : vocabulary size for time-gap embedding
    dropout          : dropout rate
    pretrained_item_emb : optional (num_items, d) numpy array for item init
    """

    def __init__(self,
                 num_items: int,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 max_seq_len: int = 200,
                 num_time_buckets: int = 256,
                 dropout: float = 0.2,
                 pretrained_item_emb: np.ndarray = None):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len

        # ── Embedding tables ─────────────────────────────────────────────────
        # Item embedding: index 0 = PAD; items are 1-indexed
        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        if pretrained_item_emb is not None:
            # pretrained_item_emb is 0-indexed (index i → item i+1)
            with torch.no_grad():
                self.item_emb.weight[1:] = torch.tensor(
                    pretrained_item_emb, dtype=torch.float)

        # Rating embedding: 0=PAD, 1-5=ratings
        self.rating_emb   = nn.Embedding(6, embed_dim, padding_idx=0)

        # Time-gap embedding: 0=PAD, 1..num_time_buckets
        self.time_emb     = nn.Embedding(num_time_buckets + 1, embed_dim, padding_idx=0)

        # Unexpectedness projection: scalar → d-dim
        self.unexp_proj   = nn.Linear(1, embed_dim)

        # Learnable positional encoding
        self.pos_emb      = nn.Embedding(max_seq_len + 1, embed_dim, padding_idx=0)

        # ── Gated Fusor ──────────────────────────────────────────────────────
        self.fusor = GatedFusor(embed_dim, dropout)

        # ── Transformer Blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            NOVATransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

        self._init_weights()
        self.register_buffer('_causal_mask_cache', torch.empty(0), persistent=False)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── Causal mask ──────────────────────────────────────────────────────────

    def _get_causal_mask(self, T: int, device) -> torch.Tensor:
        if self._causal_mask_cache.shape == torch.Size([T, T]):
            return self._causal_mask_cache
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        self._causal_mask_cache = mask.to(device)
        return self._causal_mask_cache

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self,
                seq_iid:   torch.Tensor,   # (B, T) item ids, 1-indexed, 0=pad
                seq_rat:   torch.Tensor,   # (B, T) ratings  0-5
                seq_tgap:  torch.Tensor,   # (B, T) time-gap buckets
                seq_unexp: torch.Tensor,   # (B, T) unexp scores (float)
                pad_mask:  torch.Tensor    # (B, T) bool True=real
               ) -> torch.Tensor:
        """Returns (B, T, d) hidden states."""
        B, T = seq_iid.shape

        # Positional ids: 1..T for real, 0 for padded
        pos_ids = torch.arange(1, T + 1, device=seq_iid.device).unsqueeze(0)
        pos_ids = pos_ids * pad_mask.long()

        # ── Feature embeddings ───────────────────────────────────────────────
        f_item   = self.item_emb(seq_iid)                        # (B,T,d)
        f_rating = self.rating_emb(seq_rat)                      # (B,T,d)
        f_time   = self.time_emb(seq_tgap)                       # (B,T,d)
        f_unexp  = self.unexp_proj(seq_unexp.unsqueeze(-1))      # (B,T,d)
        pos      = self.pos_emb(pos_ids)                         # (B,T,d)

        # Add positional info to item embedding before fusion
        f_item = f_item + pos
        f_item = self.dropout(f_item)

        # ── Gated Fusion → R ─────────────────────────────────────────────────
        R = self.fusor(f_item, f_rating, f_time, f_unexp)       # (B,T,d)

        # ── Causal mask ──────────────────────────────────────────────────────
        causal = self._get_causal_mask(T, seq_iid.device)

        # ── Transformer blocks ───────────────────────────────────────────────
        # f_item stays pure (used as V in every block)
        for block in self.blocks:
            R = block(R, f_item, causal, pad_mask)

        return self.final_norm(R)   # (B, T, d)

    # ── Score a set of candidate items ───────────────────────────────────────

    def score_candidates(self,
                         hidden: torch.Tensor,        # (B, d)  last real position
                         candidate_iids: torch.Tensor # (B, C)  candidate item ids
                        ) -> torch.Tensor:
        """
        Returns (B, C) logit scores.
        hidden: representation of the last real position in each sequence.
        """
        cand_emb = self.item_emb(candidate_iids)       # (B, C, d)
        scores   = torch.bmm(cand_emb, hidden.unsqueeze(-1)).squeeze(-1)  # (B,C)
        return scores

    # ── Extract last-position hidden state ───────────────────────────────────

    @staticmethod
    def gather_last(hidden: torch.Tensor,
                    pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, d): the hidden state at the last real (non-padded) position.
        """
        # last real index per sample
        lengths  = pad_mask.long().sum(dim=1) - 1          # (B,)
        lengths  = lengths.clamp(min=0)
        idx      = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        last     = hidden.gather(1, idx).squeeze(1)         # (B, d)
        return last
