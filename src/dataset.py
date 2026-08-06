from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.preprocess import Vocabulary, tokenize


class GrammarCorrectionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabulary: Vocabulary,
        maximum_length: int = 80,
    ) -> None:
        required_columns = {"incorrect", "corrected"}
        missing_columns = required_columns.difference(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if maximum_length < 3:
            raise ValueError("maximum_length must be at least 3.")

        self.dataframe = dataframe.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.maximum_length = maximum_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]

        source_tokens = tokenize(row["incorrect"])
        target_tokens = tokenize(row["corrected"])

        # Reserve one position for EOS.
        source_tokens = source_tokens[: self.maximum_length - 1]

        # Reserve two positions for SOS and EOS.
        target_tokens = target_tokens[: self.maximum_length - 2]

        source_indices = self.vocabulary.encode(
            source_tokens,
            add_sos=False,
            add_eos=True,
        )

        target_indices = self.vocabulary.encode(
            target_tokens,
            add_sos=True,
            add_eos=True,
        )

        return (
            torch.tensor(source_indices, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long),
        )


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_sequences, target_sequences = zip(*batch)

    source_lengths = torch.tensor(
        [len(sequence) for sequence in source_sequences],
        dtype=torch.long,
    )

    sorted_indices = torch.argsort(
        source_lengths,
        descending=True,
    )

    source_lengths = source_lengths[sorted_indices]

    source_sequences = [
        source_sequences[index]
        for index in sorted_indices.tolist()
    ]

    target_sequences = [
        target_sequences[index]
        for index in sorted_indices.tolist()
    ]

    source_batch = pad_sequence(
        source_sequences,
        batch_first=False,
        padding_value=pad_index,
    )

    target_batch = pad_sequence(
        target_sequences,
        batch_first=False,
        padding_value=pad_index,
    )

    return source_batch, source_lengths, target_batch


def create_dataloader(
    dataset: GrammarCorrectionDataset,
    batch_size: int,
    shuffle: bool,
    pad_index: int,
    seed: int = 468,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(
            collate_batch,
            pad_index=pad_index,
        ),
        generator=generator if shuffle else None,
    )


def load_splits(
    data_directory: str | Path = "data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = Path(data_directory)

    train = pd.read_csv(directory / "train.csv")
    validation = pd.read_csv(directory / "val.csv")
    test = pd.read_csv(directory / "test.csv")

    return train, validation, test


def build_data_pipeline(
    data_directory: str | Path = "data/processed",
    minimum_frequency: int = 2,
    maximum_vocabulary_size: int = 30_000,
    maximum_length: int = 80,
    batch_size: int = 32,
    seed: int = 468,
) -> dict[str, object]:
    train_frame, validation_frame, test_frame = load_splits(
        data_directory
    )

    vocabulary = Vocabulary(
        minimum_frequency=minimum_frequency,
        maximum_size=maximum_vocabulary_size,
    )

    combined_training_sentences = pd.concat(
        [
            train_frame["incorrect"],
            train_frame["corrected"],
        ],
        ignore_index=True,
    )

    vocabulary.build(
        tokenize(sentence)
        for sentence in combined_training_sentences
    )

    train_dataset = GrammarCorrectionDataset(
        train_frame,
        vocabulary,
        maximum_length,
    )

    validation_dataset = GrammarCorrectionDataset(
        validation_frame,
        vocabulary,
        maximum_length,
    )

    test_dataset = GrammarCorrectionDataset(
        test_frame,
        vocabulary,
        maximum_length,
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pad_index=vocabulary.pad_index,
        seed=seed,
    )

    validation_loader = create_dataloader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pad_index=vocabulary.pad_index,
        seed=seed,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pad_index=vocabulary.pad_index,
        seed=seed,
    )

    vocabulary.save(
        Path(data_directory) / "vocabulary.json"
    )

    return {
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "vocabulary": vocabulary,
    }