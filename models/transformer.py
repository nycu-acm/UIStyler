import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from typing import Optional, List
from einops.layers.torch import Rearrange
from einops import repeat, rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        image_height = image_width = img_size
        patch_height = patch_width = patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x

class SourceEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Patch embedding module.
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size)

        # Learnable positional embedding of shape [num_patches, 1, d_model]
        self.pos_embedding = nn.Parameter(torch.randn(self.patch_embed.num_patches, 1, d_model))

        # Multiple encoder layers
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.source_encoder = _get_clones(encoder_layer, num_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_img):
        ### Patch embedding ###
        source_embeds = self.patch_embed(source_img)
        source_embeds = source_embeds.flatten(2).permute(2, 0, 1)

        ### Add positional embedding ###
        source_embeds = source_embeds + self.pos_embedding

        ### Feature extraction ###
        source_feats = source_embeds
        for layer in self.source_encoder:
            source_feats = source_feats + layer(source_feats)

        return source_feats

class TargetEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.target_encoder = _get_clones(encoder_layer, num_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, target_img):
        ### Patch embedding ###
        target_embeds = self.patch_embed(target_img)
        target_embeds = target_embeds.flatten(2).permute(2, 0, 1)

        ### Feature extraction ###
        target_feats = target_embeds
        for layer in self.target_encoder:
            target_feats = layer(target_feats)

        return target_feats

class StyleTrans(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = _get_clones(decoder_layer, num_layers)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_feats, target_feats, pos):
        ### Fusion of source and target features ###
        hs = source_feats
        for layer in self.decoder:
            hs = layer(hs, target_feats, pos=pos)

        ### HWxNxC to NxCxHxW to
        N, B, C = hs.shape
        H = int(np.sqrt(N))
        assert H * H == N, "Number of patches must be a perfect square"
        hs = hs.permute(1, 2, 0).view(B, C, H, H)

        return hs

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
            )

    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # Attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_att1 = nn.Dropout(dropout)
        self.norm_att = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout_ffn = nn.Dropout(dropout)        
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, pos=None):
        # Attention
        attn_output = self.self_attn(query=src,
                                    key=src, value=src)[0]
        src = self.norm_att(self.dropout_att1(attn_output) + src)

        # Feedforward
        ffn_output = self.ffn(src)
        src = self.norm_ffn(self.dropout_ffn(ffn_output) + src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # Attention mechanism
        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_att1 = nn.Dropout(dropout)
        self.norm_att1 = nn.LayerNorm(d_model)

        # self.self_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.dropout_att2 = nn.Dropout(dropout)
        # self.norm_att2 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout_ffn = nn.Dropout(dropout)      
        self.norm_ffn = nn.LayerNorm(d_model)

        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, content, style, pos=None):
        # Attention 1
        attn_output = self.self_attn_1(query=content,
                                    key=self.with_pos_embed(style, pos), value=style)[0]
        content = self.norm_att1(self.dropout_att1(attn_output) + content)

        # # Attention 2
        # attn_output = self.self_attn_2(query=content,
        #                             key=self.with_pos_embed(style, pos), value=style)[0]
        # content = self.norm_att2(self.dropout_att2(attn_output) + content)

        # Feedforward
        ffn_output = self.ffn(content)
        content = self.norm_ffn(self.dropout_ffn(ffn_output) + content)

        return content

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
