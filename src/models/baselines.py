"""
Competitive MIL Baselines for fair comparison:
1. ABMIL (Ilse et al., ICML 2018)
2. TransMIL (Shao et al., NeurIPS 2021)
3. CLAM-SB (Lu et al., Nature BME 2021)
4. ViT-Slice-MLP (Deep single-slice MLP baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModalityFusion(nn.Module):
    """Concatenates 4 modality features (4 x 384 = 1536) and projects to d_model."""
    def __init__(self, d_in: int = 384, d_model: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask):
        x = torch.cat([t1, t1ce, t2, flair], dim=-1)
        return self.proj(x)

class ABMIL(nn.Module):
    """Attention-Based MIL with Gated Attention (Ilse et al. 2018)."""
    def __init__(self, d_in: int = 384, d_model: int = 128, d_attn: int = 64):
        super().__init__()
        self.fusion = SimpleModalityFusion(d_in, d_model)
        self.attn_V = nn.Sequential(nn.Linear(d_model, d_attn), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(d_model, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, **kwargs):
        h = self.fusion(t1, t1ce, t2, flair, seq_mask)
        a_V = self.attn_V(h)
        a_U = self.attn_U(h)
        a = self.attn_w(a_V * a_U).squeeze(-1)
        a = a.masked_fill(~seq_mask, -1e9)
        a = torch.softmax(a, dim=1)
        bag = (h * a.unsqueeze(-1)).sum(dim=1)
        return self.classifier(bag)

class PPEG(nn.Module):
    """1D Pyramid Position Encoding Generator for TransMIL."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, 7, padding=3, groups=d_model)

    def forward(self, x):
        cls_token = x[:, :1, :]
        feat = x[:, 1:, :]
        xt = feat.transpose(1, 2)
        x1 = self.conv1(xt).transpose(1, 2)
        x2 = self.conv2(xt).transpose(1, 2)
        x3 = self.conv3(xt).transpose(1, 2)
        feat = feat + x1 + x2 + x3
        return torch.cat([cls_token, feat], dim=1)

class TransMIL(nn.Module):
    """TransMIL with CLS token, self-attention, and PPEG (Shao et al. 2021)."""
    def __init__(self, d_in: int = 384, d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.fusion = SimpleModalityFusion(d_in, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ppeg = PPEG(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.10, batch_first=True, activation="gelu",
        )
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, **kwargs):
        h = self.fusion(t1, t1ce, t2, flair, seq_mask)
        B = h.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=h.device)
        full_mask = torch.cat([cls_mask, seq_mask], dim=1)
        h = self.encoder1(h, src_key_padding_mask=~full_mask)
        h = self.ppeg(h)
        h = self.encoder1(h, src_key_padding_mask=~full_mask)
        h = self.norm(h)
        return self.classifier(h[:, 0, :])

class CLAM_SB(nn.Module):
    """CLAM Single-Branch with instance-level clustering constraint (Lu et al. 2021)."""
    def __init__(self, d_in: int = 384, d_model: int = 128, d_attn: int = 64, n_clusters: int = 8):
        super().__init__()
        self.fusion = SimpleModalityFusion(d_in, d_model)
        self.attn_V = nn.Sequential(nn.Linear(d_model, d_attn), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(d_model, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1)
        self.inst_classifier = nn.Sequential(nn.Linear(d_model, 2))
        self.n_clusters = n_clusters
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, return_inst_loss=False, center_label=None, **kwargs):
        h = self.fusion(t1, t1ce, t2, flair, seq_mask)
        a_V = self.attn_V(h)
        a_U = self.attn_U(h)
        a = self.attn_w(a_V * a_U).squeeze(-1)
        a = a.masked_fill(~seq_mask, -1e9)
        a_weights = torch.softmax(a, dim=1)
        bag = (h * a_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(bag)

        inst_loss = torch.tensor(0.0, device=h.device)
        if return_inst_loss and center_label is not None:
            for b_idx in range(h.size(0)):
                valid = seq_mask[b_idx]
                if valid.sum() == 0:
                    continue
                valid_h = h[b_idx, valid]
                valid_a = a_weights[b_idx, valid]
                k = min(self.n_clusters, valid_h.size(0))
                topk_idx = torch.topk(valid_a, k).indices
                inst_logits = self.inst_classifier(valid_h[topk_idx])
                inst_labels = torch.full((k,), int(center_label[b_idx].item()),
                                         dtype=torch.long, device=h.device)
                inst_loss = inst_loss + F.cross_entropy(inst_logits, inst_labels)
            inst_loss = inst_loss / max(h.size(0), 1)

        if return_inst_loss:
            return logits, inst_loss
        return logits

class ViTSliceMLP(nn.Module):
    """Deep single-slice MLP baseline (no axial window context)."""
    def __init__(self, d_in: int = 384, d_hidden: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_in * 4, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_hidden // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, center_pos=None, **kwargs):
        if center_pos is not None:
            # Extract center slice
            B = t1.size(0)
            c = center_pos[0].item() if isinstance(center_pos, torch.Tensor) else int(center_pos)
            s_t1 = t1[:, c, :]
            s_t1ce = t1ce[:, c, :]
            s_t2 = t2[:, c, :]
            s_flair = flair[:, c, :]
        else:
            mid = t1.size(1) // 2
            s_t1 = t1[:, mid, :]
            s_t1ce = t1ce[:, mid, :]
            s_t2 = t2[:, mid, :]
            s_flair = flair[:, mid, :]

        x = torch.cat([s_t1, s_t1ce, s_t2, s_flair], dim=-1)
        return self.classifier(x)
