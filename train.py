import os
import argparse
from typing import Dict, Tuple, Optional

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
    parser = argparse.ArgumentParser(
        description="Train Qwen3 Dense or MoE model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["dense", "moe"]
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="neifuisan/Neuro-sama-QnA"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2-0.5B"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    return parser.parse_args()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_text_fields(sample: Dict) -> Tuple[str, str]:

    instruction = str(sample.get("instruction", "")).strip()
    user_input = str(sample.get("input", "")).strip()
    output = str(sample.get("output", "")).strip()

    if not instruction or not output:
        return "", ""

    if user_input:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    return prompt, output


class QnADataset(Dataset):

    def __init__(self, hf_dataset):

        self.examples = []

        for sample in hf_dataset:

            prompt, response = pick_text_fields(sample)

            if prompt and response:

                self.examples.append(
                    {
                        "prompt": prompt,
                        "response": response
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_seq_len):

    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]

    eos = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    full_texts = [
        prompt + response + eos
        for prompt, response in zip(prompts, responses)
    ]

    full_enc = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt"
    )

    prompt_enc = tokenizer(
        prompts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    labels = input_ids.clone()

    prompt_lengths = prompt_enc["attention_mask"].sum(dim=1)

    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100

    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def build_model(model_type: str, config: Config):

    if model_type == "dense":
        return Qwen3DenseLM(config)

    elif model_type == "moe":
        return Qwen3MOELM(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    save_dir,
    model_type,
    epoch,
    scaler=None,
    grad_clip=1.0,
    start_batch=0,
):

    model.train()

    total_loss = 0.0
    total_aux = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}",
        leave=False
    )

    for batch_idx, batch in enumerate(progress_bar):

        if batch_idx < start_batch:
            continue

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:

            with autocast(device_type="cuda", dtype=torch.float16):

                outputs = model(
                    input_ids=input_ids,
                    labels=labels
                )

                loss = outputs["loss"]

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            outputs = model(
                input_ids=input_ids,
                labels=labels
            )

            loss = outputs["loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

            optimizer.step()

        total_loss += loss.item()

        total_aux += float(
            outputs.get("aux_loss", 0.0)
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

        latest_checkpoint = os.path.join(
            save_dir,
            f"latest_{model_type}.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "batch": batch_idx,
            },
            latest_checkpoint,
        )

    avg_loss = total_loss / max(1, len(dataloader))
    avg_aux = total_aux / max(1, len(dataloader))

    return avg_loss, avg_aux

@torch.no_grad()
def evaluate(model, dataloader, device, scaler=None):

    model.eval()

    total_loss = 0.0
    total_aux = 0.0

    for batch in tqdm(
        dataloader,
        desc="Validation",
        leave=False
    ):

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if scaler is not None:

            with autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    input_ids=input_ids,
                    labels=labels
                )

                loss = outputs["loss"]

        else:

            outputs = model(
                input_ids=input_ids,
                labels=labels
            )

            loss = outputs["loss"]

        total_loss += loss.item()

        total_aux += float(
            outputs.get("aux_loss", 0.0)
        )

    avg_loss = total_loss / max(1, len(dataloader))
    avg_aux = total_aux / max(1, len(dataloader))

    return avg_loss, avg_aux


def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")

    raw_dataset = load_dataset(args.dataset)

    if "train" not in raw_dataset:
        raise ValueError(
            "Dataset must contain a train split."
        )

    full_train_split = raw_dataset["train"]

    split = full_train_split.train_test_split(
        test_size=args.val_split_ratio,
        seed=args.seed
    )

    train_split = split["train"]
    val_split = split["test"]

    config = Config()

    config.vocab_size = len(tokenizer)

    train_dataset = QnADataset(train_split)
    val_dataset = QnADataset(val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch:
            collate_fn(
                batch,
                tokenizer,
                config.max_seq_len
            )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch:
            collate_fn(
                batch,
                tokenizer,
                config.max_seq_len
            )
    )

    print(f"Building {args.model} model...")

    model = build_model(
        args.model,
        config
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=0.1,
        weight_decay=config.weight_decay
    )

    scaler = (
        GradScaler("cuda")
        if device.type == "cuda"
        else None
    )
    start_epoch = 0
    start_batch = 0
    best_val_loss = float("inf")

    if args.resume is not None:

        print(
            f"Loading checkpoint: {args.resume}"
        )
        checkpoint = torch.load(
            args.resume,
            map_location=device
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        start_epoch = checkpoint.get(
            "epoch",
            0
        )

        start_batch = checkpoint.get(
            "batch",
            0
        ) + 1

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        print(
            f"Resuming from "
            f"Epoch {start_epoch+1}, "
            f"Batch {start_batch}"
        )

    print(f"Training on {device}...")
    
    try:

        for epoch in range(start_epoch, args.epochs):

            train_loss, train_aux = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                save_dir=args.save_dir,
                model_type=args.model,
                epoch=epoch,
                scaler=scaler,
                grad_clip=1.0,
                start_batch=start_batch if epoch == start_epoch else 0,
            )

            start_batch = 0

            val_loss, val_aux = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                scaler=scaler
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"train_aux: {train_aux:.4f} | "
                f"val_aux: {val_aux:.4f}"
            )

            epoch_path = os.path.join(
                args.save_dir,
                f"epoch_{epoch+1}_{args.model}.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "model_type": args.model,
                    "tokenizer": args.tokenizer,
                    "epoch": epoch + 1,
                    "batch": 0,
                    "best_val_loss": best_val_loss,
                },
                epoch_path,
            )

            print(f"Saved epoch checkpoint -> {epoch_path}")

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_path = os.path.join(
                    args.save_dir,
                    f"best_{args.model}.pt"
                )

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "model_type": args.model,
                        "tokenizer": args.tokenizer,
                        "epoch": epoch + 1,
                        "batch": 0,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                    },
                    best_path,
                )

                print(f"Saved best checkpoint -> {best_path}")

    except KeyboardInterrupt:

        interrupt_path = os.path.join(
            args.save_dir,
            f"interrupted_{args.model}.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "model_type": args.model,
                "tokenizer": args.tokenizer,
                "epoch": epoch,
                "batch": start_batch,
                "best_val_loss": best_val_loss,
            },
            interrupt_path,
        )

        print(
            f"\nTraining interrupted.\n"
            f"Checkpoint saved to {interrupt_path}"
        )

        return

    final_path = os.path.join(
        args.save_dir,
        f"final_{args.model}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "model_type": args.model,
            "tokenizer": args.tokenizer,
            "best_val_loss": best_val_loss,
            "epoch": args.epochs,
            "batch": 0,
        },
        final_path,
    )

    print(f"Saved final checkpoint -> {final_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()