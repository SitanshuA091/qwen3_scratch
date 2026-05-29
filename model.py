import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import math
import random
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer

from dataclasses import dataclass
from typing import List, Optional

import warnings
import os

@dataclass
class Config:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    n_kv_heads: int = 4

    vocab_size: int = 32000
    max_seq_len: int = 512
    sliding_window: int = 4096

    batch_size: int = 24
    max_steps: int = 500
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True

    attn_bias: bool = False
    rms_norm_eps: float = 1e-6
    
    n_experts: int = 128
    n_active_experts: int = 8
    moe_expert_hidden_dim: int = 1536
    moe_aux_loss_coef: float = 0.01
    tie_embeddings: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

        self.d_k = self.d_model // self.n_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

# final output proj
class OutputHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [seq_len, dim//2]

        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # [seq_len, dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        assert seq_len <= self.cos.size(0)

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0) 
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        x = x.to(torch.float32)
        out = (x * cos) + (self.rotate_half(x) * sin)
        return out.type_as(x)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attn_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attn_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attn_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.q_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)

        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)  # [B, S, H*d_k]
        k = self.k_proj(x)  # [B, S, KV*d_k]
        v = self.v_proj(x)  # [B, S, KV*d_k]

        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # [B, S, H, D] -> [B, H, S, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = self.rotary(q)
        k = self.rotary(k)

        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        # [B, H, S, D] -> [B, S, d_model]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)
    
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(activated_x))
    
class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # FFN
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x
    
class Qwen3DenseLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
       

        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

        return {
            "logits": logits,
            "loss": loss
        }
        
def shift_inputs_and_labels(input_ids):
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    return x, y