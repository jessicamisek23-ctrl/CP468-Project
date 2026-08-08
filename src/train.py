from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import build_data_pipeline
from src.seq2seq import build_model
from src.utils import set_seed, count_parameters

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
):
    model.train()

    epoch_loss = 0

    for source, lengths, target in dataloader:

        source = source.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            source,
            lengths,
            target,
            teacher_forcing_ratio=0.5,
        )

        outputs = outputs[1:].reshape(
            -1,
            outputs.shape[-1],
        )

        targets = target[1:].reshape(-1)

        loss = criterion(
            outputs,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
):
    model.eval()

    epoch_loss = 0

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

        outputs = outputs[1:].reshape(
            -1,
            outputs.shape[-1],
        )

        targets = target[1:].reshape(-1)

        loss = criterion(
            outputs,
            targets,
        )

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def main():
    
    set_seed(468)

    pipeline = build_data_pipeline()

    vocab = pipeline["vocabulary"]

    train_loader = pipeline["train_loader"]
    validation_loader = pipeline["validation_loader"]

    model = build_model(
        vocabulary_size=len(vocab),
        pad_index=vocab.pad_index,
        sos_index=vocab.sos_index,
        eos_index=vocab.eos_index,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_index
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    best_val_loss = float("inf")

    Path("models").mkdir(exist_ok=True)

    epochs = 20

    for epoch in range(epochs):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss = validate(
            model,
            validation_loader,
            criterion,
        )

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "models/best_model.pt",
            )

            print("Saved best model.")

    print("Training complete.")


if __name__ == "__main__":
    main()