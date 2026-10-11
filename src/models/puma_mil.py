"""
PUMA-MIL: Prototype-Updated Multi-Modal Attention-Based Multiple Instance Learning.

Architecture:
1. Reliability-weighted modality fusion with stochastic modality dropout.
2. 2-layer Transformer encoder with learnable positional embeddings.
3. Slice-level primary classification head on center slice.
4. Hybrid MIL pooling (gated attention modulated by entropy reliability + top-K).
5. Exponential Moving Average (EMA) class prototypes with cosine similarity anchoring.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_topk_mean(seq_feats: torch.Tensor, scores: torch.Tensor, seq_mask: torch.Tensor, top_k: int) -> torch.Tensor:
    """Computes the mean feature of the top-k highest scoring valid instances."""
    bsz, _, dim = seq_feats.shape
    out = []
    device = seq_feats.device
    dtype = seq_feats.dtype

    for b in range(bsz):
        valid_idx = torch.where(seq_mask[b])[0]
        if valid_idx.numel() == 0:
            out.append(torch.zeros(dim, device=device, dtype=dtype))
            continue
        vals = scores[b, valid_idx]
        k = min(int(top_k), int(valid_idx.numel()))
        chosen_local = torch.topk(vals, k=k, dim=0).indices
        chosen_idx = valid_idx[chosen_local]
        out.append(seq_feats[b, chosen_idx].mean(dim=0))
    return torch.stack(out, dim=0)

class ReliabilityFusion(nn.Module):
    """
    Reliability-Weighted Modality Fusion Head.
    Projects 4 modality feature streams (T1, T1ce, T2, FLAIR) to shared dimension,
    computes learned reliability scalar scores, normalizes via masked softmax,
    and fuses weighted representations with mean features.
    """
    def __init__(self, d_in: int = 384, d_model: int = 128):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            ) for _ in range(4)
        ])
        self.rel = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, mods, mod_mask):
        proj_feats = []
        rel_logits = []

        for i in range(4):
            p = self.proj[i](mods[i])
            proj_feats.append(p)
            rel_logits.append(self.rel[i](p))

        feat_stack = torch.stack(proj_feats, dim=2)    # [B, W, 4, D]
        rel_stack = torch.cat(rel_logits, dim=-1)      # [B, W, 4]
        rel_stack = rel_stack.masked_fill(~mod_mask[:, None, :], -1e4)

        weights = torch.softmax(rel_stack, dim=-1)
        fused = (feat_stack * weights.unsqueeze(-1)).sum(dim=2)

        valid = mod_mask[:, None, :, None].float()
        mean_feat = (feat_stack * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)

        out = self.out(torch.cat([fused, mean_feat], dim=-1))
        return out, weights

class HybridMILPool(nn.Module):
    """
    Hybrid MIL Pooling:
    Combines learned gated attention (modulated by entropy-based prediction reliability)
    with deterministic top-K positive and negative instance features.
    """
    def __init__(self, d_model: int = 128, top_k: int = 4):
        super().__init__()
        self.top_k = top_k
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, seq_feats, slice_logits, seq_mask):
        probs = torch.softmax(slice_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1) / math.log(2.0)
        reliability = 1.0 - entropy

        attn_logits = self.attn(seq_feats).squeeze(-1)
        attn_logits = attn_logits + torch.log(reliability.clamp_min(1e-6))
        attn_logits = attn_logits.masked_fill(~seq_mask, -1e9)

        attn_w = torch.softmax(attn_logits, dim=1)
        attn_pool = (seq_feats * attn_w.unsqueeze(-1)).sum(dim=1)

        pos_feat = masked_topk_mean(seq_feats, probs[..., 1], seq_mask, self.top_k)
        neg_feat = masked_topk_mean(seq_feats, probs[..., 0], seq_mask, self.top_k)

        bag_embed = torch.cat([attn_pool, pos_feat, neg_feat], dim=-1)  # [B, 3*d_model]
        return bag_embed, attn_w, reliability

class PUMAMILLite(nn.Module):
    """
    PUMA-MIL Architecture.
    """
    def __init__(self, d_in: int = 384, d_model: int = 128, max_slices: int = 48,
                 top_k: int = 4, proto_momentum: float = 0.90, modality_dropout: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.max_slices = max_slices
        self.top_k = top_k
        self.proto_momentum = proto_momentum
        self.modality_dropout_prob = modality_dropout

        self.fusion = ReliabilityFusion(d_in=d_in, d_model=d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.slice_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2)
        )

        self.pool = HybridMILPool(d_model=d_model, top_k=top_k)

        self.register_buffer("prototypes", torch.zeros(2, d_model * 3))
        self.register_buffer("proto_counts", torch.zeros(2))

        self.final_head = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 2),
            nn.Linear(d_model * 3 + 2, d_model),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model, 2)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def apply_modality_dropout(self, mod_mask: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.modality_dropout_prob <= 0:
            return mod_mask

        keep = mod_mask.clone()
        rand = torch.rand_like(keep.float())
        keep = keep & (rand > self.modality_dropout_prob)

        # Ensure at least one modality remains active
        for i in range(keep.size(0)):
            if keep[i].sum() == 0:
                valid = torch.where(mod_mask[i])[0]
                if valid.numel() > 0:
                    j = valid[torch.randint(0, valid.numel(), (1,), device=valid.device)]
                    keep[i, j] = True
                else:
                    keep[i] = True

        return keep

    def update_prototypes(self, bag_embed: torch.Tensor, bag_labels: torch.Tensor):
        with torch.no_grad():
            for c in [0, 1]:
                mask = (bag_labels == c)
                if mask.any():
                    batch_proto = bag_embed[mask].mean(dim=0)
                    if self.proto_counts[c] == 0:
                        self.prototypes[c] = batch_proto
                    else:
                        self.prototypes[c] = (
                            self.proto_momentum * self.prototypes[c] +
                            (1.0 - self.proto_momentum) * batch_proto
                        )
                    self.proto_counts[c] += mask.sum()

    def prototype_sims(self, bag_embed: torch.Tensor) -> torch.Tensor:
        sims = []
        for c in [0, 1]:
            if self.proto_counts[c] <= 0 or torch.norm(self.prototypes[c]) < 1e-8:
                sims.append(torch.zeros(bag_embed.size(0), 1, device=bag_embed.device))
            else:
                s = F.cosine_similarity(
                    bag_embed, self.prototypes[c].unsqueeze(0), dim=-1
                ).unsqueeze(-1)
                sims.append(s)
        return torch.cat(sims, dim=-1)

    def forward(self, t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=False):
        assert t1.size(1) <= self.max_slices, "Input sequence longer than model max_slices"

        mod_mask_used = self.apply_modality_dropout(mod_mask)
        fused, mod_weights = self.fusion([t1, t1ce, t2, flair], mod_mask_used)

        x = fused + self.pos_embed[:, :t1.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~seq_mask)

        slice_logits = self.slice_head(x)
        bag_embed, attn_w, reliability = self.pool(x, slice_logits, seq_mask)
        proto_s = self.prototype_sims(bag_embed)
        bag_logits = self.final_head(torch.cat([bag_embed, proto_s], dim=-1))

        if return_aux:
            return bag_logits, slice_logits, bag_embed, mod_weights, attn_w, reliability
        return bag_logits
