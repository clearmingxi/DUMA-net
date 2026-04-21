import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
import numpy as np
from loss import ClipLoss

class CoOccurrenceBlock(nn.Module):
    """
    NCT-C variant of the Neuro-Cognitive Transformer.
    A shallow module with learnable co-occurrence slot embeddings.
    """
    def __init__(self, embed_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # multiple attention heads
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # learnable positional encoding for slots
        self.freq_position_encoding = nn.Parameter(
            torch.randn(1, 100, embed_dim) * 0.02
        )

        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attention=False):
        seq_len = x.shape[1]
        if seq_len <= self.freq_position_encoding.shape[1]:
            pos_encoding = self.freq_position_encoding[:, :seq_len, :]
            x = x + pos_encoding

        attn_out, attn_weights = self.multihead_attention(
            x, x, x, need_weights=return_attention, average_attn_weights=False
        )
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attn_weights
        else:
            return x


class NCT_Base(nn.Module):
    """
    Improved version of EfficientEEGEncoder, serving as the base encoder
    under the Neuro-Cognitive Transformer (NCT) framework.
    Use configuration flags to enable either NCT-S or NCT-C modules.
    """
    def __init__(self, config, input_shape=(63, 250),
                 d_model=1024, num_subjects=10):
        super().__init__()
        self.config = config
        n_channels, seq_length = input_shape

        # subject embedding for personalization
        self.subject_embedding = nn.Embedding(num_subjects, d_model)

        # strong patch embedding network
        self.patch_embedding = self._build_strong_patch_embedding(
            self.config.emb_size)

        # NCT-S: canonical sequential transformer
        if self.config.use_ncts:
            self.ncts_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.config.emb_size,
                    nhead=4,
                    dim_feedforward=self.config.emb_size * 2,
                    dropout=self.config.dropout * 0.5,
                    batch_first=True
                ) for _ in range(2)
            ])

        # NCT-C: shallow co-occurrence slot model
        if self.config.use_nctc:
            self.nctc_model = CoOccurrenceBlock(
                self.config.emb_size,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout * 0.5
            )

        self.flatten = nn.Flatten()
        flat_dim = self.config.emb_size * 50

        self.projection = nn.Sequential(
            nn.Linear(flat_dim, d_model * 2),
            nn.LayerNorm(d_model * 2), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(self.config.dropout * 0.5),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model)
        )

 
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))
        self.loss_func = ClipLoss()
        self._init_weights()

    def _build_strong_patch_embedding(self, emb_size):
        return nn.Sequential(
            nn.Conv2d(1, 32, (1, 25), stride=(1, 1), padding=(0, 12)),
            nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 5), (1, 5)),
            nn.Conv2d(32, 64, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(64), nn.ELU(), nn.Dropout(0.3),
            nn.Conv2d(64, emb_size, (1, 1)), nn.BatchNorm2d(emb_size), nn.ELU(),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x, subject_ids=None,
                return_dict=False, return_attention=False):
        # collapse channel dimension if present
        if len(x.shape) == 4:
            x = x.mean(dim=1) if x.shape[1] > 1 else x.squeeze(1)
        x = x.unsqueeze(1)

        patch_embeds = self.patch_embedding(x)

        final_features = patch_embeds
        freq_attn = None

        if self.config.use_ncts:
            for layer in self.ncts_layers:
                final_features = layer(final_features)
        elif self.config.use_nctc:
            if return_attention:
                final_features, freq_attn = self.nctc_model(
                    patch_embeds, return_attention=True)
            else:
                final_features = self.nctc_model(
                    patch_embeds, return_attention=False)

        x = self.flatten(final_features)
        embeddings = self.projection(x)
        if subject_ids is not None:
            embeddings = embeddings + self.subject_embedding(subject_ids)

        if return_attention:
            return embeddings, freq_attn
        elif return_dict:
            return {"embeddings": embeddings}
        else:
            return embeddings

# backward-compatible aliases
EfficientEEGEncoderV2 = NCT_Base
NCTCTransformer = CoOccurrenceBlock
