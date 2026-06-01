import os
import argparse
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from configs import Config
from model import Qwen3DenseLM, Qwen3MOELM

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3 dense or MoE model")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["dense", "moe"],
        help="Choose which model to train"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="neifuisan/Neuro-sama-QnA",
        help="Hugging Face dataset name"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Hugging Face tokenizer name"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0
    )

    return parser.parse_args()

def pick_text_fields(sample: Dict) -> List[str]:
    """
    Try to infer text fields from the dataset sample.
    Works for QnA-style datasets and also generic text datasets.
    """
    common_question_keys = ["question", "prompt", "instruction", "query", "input"]
    common_answer_keys = ["answer", "response", "output", "completion", "target"]

    question = None
    answer = None

    for key in common_question_keys:
        if key in sample and sample[key] is not None:
            question = str(sample[key]).strip()
            break

    for key in common_answer_keys:
        if key in sample and sample[key] is not None:
            answer = str(sample[key]).strip()
            break

    if question and answer:
        return [f"Question: {question}\nAnswer: {answer}"]

    if "text" in sample and sample["text"] is not None:
        return [str(sample["text"]).strip()]

    # fallback: join all string-like fields
    parts = []
    for k, v in sample.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")

    if parts:
        return ["\n".join(parts)]

    return [""]


class HuggingFaceTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        texts = pick_text_fields(sample)
        text = texts[0]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(model_type: str, config: Config):
    if model_type == "dense":
        return Qwen3DenseLM(config)
    if model_type == "moe":
        return Qwen3MOELM(config)
    raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, device, scaler=None):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)

    if "train" in dataset:
        train_split = dataset["train"]
    else:
        first_split = list(dataset.keys())[0]
        train_split = dataset[first_split]

    if "validation" in dataset:
        val_split = dataset["validation"]
    elif "valid" in dataset:
        val_split = dataset["valid"]
    elif "test" in dataset:
        val_split = dataset["test"]
    else:
        val_split = None

    config = Config()
    config.max_seq_len = args.max_seq_len
    config.vocab_size = tokenizer.vocab_size

    train_dataset = HuggingFaceTextDataset(
        train_split,
        tokenizer,
        config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = None
    if val_split is not None:
        val_dataset = HuggingFaceTextDataset(
            val_split,
            tokenizer,
            config.max_seq_len
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    print(f"Building {args.model} model...")
    model = build_model(args.model, config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1
    )

    scaler = GradScaler("cuda") if device.type == "cuda" else None

    print(f"Training on {device}...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler
        )

        print(f"Epoch {epoch + 1}/{args.epochs} | train loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                scaler=scaler
            )
            print(f"Epoch {epoch + 1}/{args.epochs} | val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "model_type": args.model,
                        "tokenizer": args.tokenizer,
                    },
                    save_path
                )
                print(f"Saved best checkpoint -> {save_path}")
        else:
            save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}_{args.model}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "model_type": args.model,
                    "tokenizer": args.tokenizer,
                },
                save_path
            )
            print(f"Saved checkpoint -> {save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()