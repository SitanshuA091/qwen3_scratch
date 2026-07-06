import os
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    n_kv_heads: int = 4
    epochs :int = 5

    vocab_size: int = 32000
    max_seq_len: int = 256
    sliding_window: int = 4096

    batch_size: int = 2
    weight_decay: float = 0.1
    dropout: float = 0.1
    learning_rate: float = 1e-5
    

    attn_bias: bool = False
    rms_norm_eps: float = 1e-6
    
    #MOE specific configs
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