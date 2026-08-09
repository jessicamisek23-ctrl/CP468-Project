from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import build_data_pipeline
from src.encoder import Encoder
from src.utils import count_parameters, set_seed


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 468
EPOCHS = 20


class NoAttentionDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_index: int = 0,
    ) -> None:
        super().__init__()

        self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=pad_index,
        )

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(
                dropout if num_layers > 1 else 0
            ),
        )

        self.output_projection = nn.Linear(
            hidden_size,
            vocabulary_size,
        )

    def forward(
        self,
        input_token,
        hidden,
        cell,
    ):
        embedded = self.dropout(
            self.embedding(
                input_token.unsqueeze(0)
            )
        )

        output, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell),
        )

        logits = self.output_projection(
            output.squeeze(0)
        )

        return logits, hidden, cell


class NoAttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        pad_index,
        sos_index,
        eos_index,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    def forward(
        self,
        source,
        lengths,
        target,
        teacher_forcing_ratio=0.5,
    ):
        batch_size = source.size(1)
        target_length = target.size(0)

        outputs = torch.zeros(
            target_length,
            batch_size,
            self.decoder.vocabulary_size,
            device=source.device,
        )

        _, hidden, cell = self.encoder(
            source,
            lengths,
        )

        input_token = target[0]

        for timestep in range(
            1,
            target_length,
        ):
            logits, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell,
            )

            outputs[timestep] = logits

            use_teacher_forcing = (
                random.random()
                < teacher_forcing_ratio
            )

            prediction = logits.argmax(dim=1)

            input_token = (
                target[timestep]
                if use_teacher_forcing
                else prediction
            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        source,
        lengths,
        max_length=80,
    ):
        self.eval()

        batch_size = source.size(1)

        _, hidden, cell = self.encoder(
            source,
            lengths,
        )

        input_token = torch.full(
            (batch_size,),
            self.sos_index,
            dtype=torch.long,
            device=source.device,
        )

        predictions = []

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=source.device,
        )

        for _ in range(max_length):
            logits, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell,
            )

            prediction = logits.argmax(dim=1)

            predictions.append(prediction)

            finished |= (
                prediction == self.eos_index
            )

            input_token = prediction

            if finished.all():
                break

        return torch.stack(predictions)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
):
    model.train()

    total_loss = 0.0

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

        vocabulary_size = outputs.shape[-1]

        loss = criterion(
            outputs[1:].reshape(
                -1,
                vocabulary_size,
            ),
            target[1:].reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
):
    model.eval()

    total_loss = 0.0

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

        loss = criterion(
            outputs[1:].reshape(
                -1,
                vocabulary_size,
            ),
            target[1:].reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main() -> None:
    set_seed(SEED)

    pipeline = build_data_pipeline()

    vocabulary = pipeline["vocabulary"]
    train_loader = pipeline["train_loader"]
    validation_loader = pipeline[
        "validation_loader"
    ]

    encoder = Encoder(
        vocabulary_size=len(vocabulary),
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        pad_index=vocabulary.pad_index,
    )

    decoder = NoAttentionDecoder(
        vocabulary_size=len(vocabulary),
        embedding_dim=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.3,
        pad_index=vocabulary.pad_index,
    )

    model = NoAttentionSeq2Seq(
        encoder,
        decoder,
        vocabulary.pad_index,
        vocabulary.sos_index,
        vocabulary.eos_index,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.pad_index,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    Path("models").mkdir(exist_ok=True)
    Path("outputs/metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_loss = float("inf")
    history = []

    start = time.perf_counter()

    print(
        "No-attention parameters:",
        f"{count_parameters(model):,}",
    )

    for epoch in range(1, EPOCHS + 1):
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

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | Train: {train_loss:.4f}"
            f" | Val: {validation_loss:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            torch.save(
                model.state_dict(),
                "models/"
                "best_no_attention_model.pt",
            )

    elapsed = time.perf_counter() - start

    results = {
        "model": "LSTM without attention",
        "parameter_count": count_parameters(
            model
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "training_time_seconds": elapsed,
        "history": history,
    }

    with Path(
        "outputs/metrics/"
        "no_attention_training.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()