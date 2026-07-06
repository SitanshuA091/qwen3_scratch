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
from typing import List, Optional

from configs import Config

import warnings
import os


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(Config.vocab_size, Config.d_model)

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
    def rotate_half(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x = torch.stack(
            (-x2, x1), dim=-1
            )
        return x.flatten(-2)

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
    
def shift_inputs_and_labels(input_ids, labels):
    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]
    return input_ids, labels
    
    
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

        # weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):

        if labels is not None:
            input_ids, labels = shift_inputs_and_labels(input_ids, labels)

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
                labels.reshape(-1),
                ignore_index=-100
            )

        return {
            "logits": logits,
            "loss": loss
        }
        

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.n_active_experts = config.n_active_experts
        self.dropout = config.dropout

        hidden_dim = getattr(config, "moe_expert_hidden_dim", getattr(config, "d_ff"))

        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(self.d_model, hidden_dim, dropout=self.dropout)
            for _ in range(self.n_experts)
        ])

    def forward(self, x):
        bsz, seq_len, d_model = x.shape
        assert d_model == self.d_model

        router_logits = self.router(x)                          
        router_probs = F.softmax(router_logits, dim=-1)       

        # top-k experts per token
        topk_probs, topk_idx = torch.topk(
            router_probs, k=self.n_active_experts, dim=-1
        )                                                    

        importance = router_probs.mean(dim=(0, 1))             

        with torch.no_grad():
            load = torch.zeros(self.n_experts, device=x.device, dtype=x.dtype)
            flat_idx = topk_idx.reshape(-1)                     
            load.scatter_add_(
                0,
                flat_idx,
                torch.ones_like(flat_idx, dtype=x.dtype)
            )
            load = load / flat_idx.numel()

        aux_loss = self.n_experts * torch.sum(importance * load)

        flat_x = x.reshape(bsz * seq_len, d_model)             
        flat_topk_idx = topk_idx.reshape(-1, self.n_active_experts)
        flat_topk_probs = topk_probs.reshape(-1, self.n_active_experts)

        output = torch.zeros_like(flat_x)

        for expert_id in range(self.n_experts):
            expert_mask = (flat_topk_idx == expert_id)         
            if not expert_mask.any():
                continue

            token_mask = expert_mask.any(dim=-1)               
            token_positions = token_mask.nonzero(as_tuple=True)[0]

            expert_input = flat_x[token_positions]              
            expert_output = self.experts[expert_id](expert_input) 

            weights = (flat_topk_probs[token_positions] * expert_mask[token_positions].to(flat_topk_probs.dtype)).sum(dim=-1)
            output[token_positions] += expert_output * weights.unsqueeze(-1)

        output = output.view(bsz, seq_len, d_model)
        return output, aux_loss
    
class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.attention = GroupedQueryAttention(config)
        self.moe = MoELayer(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + self.dropout(moe_out)

        return x, aux_loss
    
class Qwen3MOELM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            MoETransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        if getattr(config, "tie_embeddings", False):
            self.lm_head.weight = self.token_embedding.weight

        self.aux_loss_coef = getattr(config, "moe_aux_loss_coef", 0.01)

    def forward(self, input_ids, labels=None):
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        total_aux_loss = 0.0

        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            loss = lm_loss + self.aux_loss_coef * total_aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss
        }