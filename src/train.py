from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import build_data_pipeline
from src.seq2seq import build_model
from src.utils import count_parameters, get_hardware_info, set_seed


SEED = 468
EPOCHS = 20
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5

EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
ATTENTION_DIM = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
) -> float:
    model.train()

    epoch_loss = 0.0

    for batch_idx, (source, lengths, target) in enumerate(dataloader):
        source = source.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            source,
            lengths,
            target,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        )

        vocabulary_size = outputs.shape[-1]

        logits = outputs[1:].reshape(
            -1,
            vocabulary_size,
        )

        targets = target[1:].reshape(-1)

        loss = criterion(logits, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        epoch_loss += loss.item()

        # Progress update every 100 batches
        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}/{len(dataloader)}"
                f" | Loss: {loss.item():.4f}"
            )

    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
) -> float:
    model.eval()

    epoch_loss = 0.0

    for source, lengths, target in dataloader:
        source = source.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target = target.to(DEVICE)

        outputs = model(
            source,
            lengths,
            target,
            teacher_forcing_ratio=0.0,
        )

        vocabulary_size = outputs.shape[-1]

        logits = outputs[1:].reshape(
            -1,
            vocabulary_size,
        )

        targets = target[1:].reshape(-1)

        loss = criterion(logits, targets)

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def main() -> None:
    set_seed(SEED)

    Path("models").mkdir(exist_ok=True)
    Path("outputs/metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline = build_data_pipeline()

    vocabulary = pipeline["vocabulary"]
    train_loader = pipeline["train_loader"]
    validation_loader = pipeline["validation_loader"]

    model = build_model(
        vocabulary_size=len(vocabulary),
        pad_index=vocabulary.pad_index,
        sos_index=vocabulary.sos_index,
        eos_index=vocabulary.eos_index,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        attention_dim=ATTENTION_DIM,
    ).to(DEVICE)

    parameter_count = count_parameters(model)
    hardware = get_hardware_info()

    print(f"Device: {DEVICE}")
    print(f"GPU: {hardware['gpu']}")
    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.pad_index,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")

    training_history = []

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
        )

        epoch_time = time.perf_counter() - epoch_start

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "time_seconds": epoch_time,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {validation_loss:.4f}"
            f" | Time: {epoch_time:.1f}s"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                "models/best_model.pt",
            )

            print("Saved best model.")

    total_training_time = (
        time.perf_counter() - start_time
    )

    results = {
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "teacher_forcing_ratio": TEACHER_FORCING_RATIO,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "attention_dim": ATTENTION_DIM,
        "vocabulary_size": len(vocabulary),
        "parameter_count": parameter_count,
        "device": str(DEVICE),
        "gpu": hardware["gpu"],
        "best_validation_loss": best_validation_loss,
        "training_time_seconds": total_training_time,
        "history": training_history,
    }

    with Path(
        "outputs/metrics/training_history.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(
        f"\nTraining complete in "
        f"{total_training_time / 60:.2f} minutes."
    )


if __name__ == "__main__":
    main()