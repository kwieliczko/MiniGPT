# SPDX-License-Identifier: Apache-2.0
# Educational implementation inspired by Sebastian Raschka's LLMs-from-scratch.
# See NOTICE for attribution and differences.
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.att = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim), nn.GELU(),
                                nn.Linear(4 * emb_dim, emb_dim))

    def forward(self, x):
        n = x.shape[1]
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        y = self.norm1(x)
        att_out, _ = self.att(y, y, y, attn_mask=mask, need_weights=False)
        x = x + att_out
        return x + self.ff(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=50257, emb_dim=64, context_length=64, n_layers=2, n_heads=4):
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(emb_dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        if seq_len > self.context_length:
            raise ValueError('Context length exceeded')
        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        return self.out_head(self.norm(self.blocks(x)))
