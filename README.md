# Tiny-Qwen

A simplified PyTorch implementation of Qwen-3 Dense and Qwen3 MoE language models.

The implementations are a scaled-down version inspired by the Qwen3-1.7B dense model and the Qwen3-30B-A3B MOE model

## Implemented Components

- Token Embeddings
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- RMSNorm
- SwiGLU Feed Forward Network
- Causal Self Attention
- Residual Connections inside transformer blocks
- Weight Tied LM Head
- Decoder-only Causal Language Modeling

---

### Dense Architecture Reference

Reference model:

```text id="n02x6i"
Qwen3-1.7B
Layers: 28
Heads: 16 / 8 KV
Context Length: 32K
```

Current scaled-down implementation:

```text id="y5kw5o"
Layers: 6
Heads: 8 / 4 KV
Context Length: 512
```

The implementation keeps the same architectural style while reducing model size and compute requirements.

---

### MoE Model

The Mixture of Experts model implementation is inspired by:

```
Qwen3-30B-A3B
Layers: 48
Heads: 32 / 4 KV
Experts: 128 total / 8 active
Context Length: 128K
```

The MoE implementation includes:

- Router / Gating Network
- Top-k Expert Routing
- Sparse Expert Activation
- Expert SwiGLU FFNs
- Auxiliary Load Balancing Loss

## Training Results and Comparisons

Training and Validation loss Results on a HF Qna dataset - neifuisan/Neuro-sama-QnA ~ 500 examples

- **Dense Model** Loss curves over training lasting 5 epochs
<img width="397.5" height="235" alt="image" src="https://github.com/user-attachments/assets/9216c125-3bd6-43d6-bc98-671afa4ab6e2" />

- **MOE Model** Loss curves over training lasting 5 epochs
<img width="397.5" height="235" alt="image" src="https://github.com/user-attachments/assets/d2f7e8f5-1d62-4771-971c-d5126e97c995" />

Training and Validation loss Results on a subset of the roneneldan/TinyStories dataset ~ 10000 samples

- **Dense Model** Loss curves over training lasting 3 epochs
<img width="393" height="235" alt="image" src="https://github.com/user-attachments/assets/8f78e446-fb5d-4357-befa-70d450daa85e" />

-  **MOE Model** Loss curves over training lasting 3 epochs
<img width="393" height="235" alt="image" src="https://github.com/user-attachments/assets/c65dff2d-ad49-4719-a754-da969507fbfa" />


## Usage

- Train the dense model:

```bash
python train.py --model dense
```

- Train the MoE model:

```bash
python train.py --model moe
```
- Train on a particular HF dataset
```bash
python train.py --model dense --dataset neifuisan/Neuro-sama-QnA
```

- Model architecture is defined in `model.py`, while hyperparameters and training configuration are managed through `configs.py`. 
