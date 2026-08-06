from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/C4_200M.tsv-00004-of-00010")
OUTPUT_DIRECTORY = Path("data/processed")

SAMPLE_SIZE = 50_000
RANDOM_SEED = 468

TRAIN_SIZE = 40_000
VALIDATION_SIZE = 5_000
TEST_SIZE = 5_000


def normalize_text(text: str) -> str:
    """Remove unnecessary surrounding and repeated whitespace."""
    return " ".join(text.strip().split())


def collect_examples() -> pd.DataFrame:
    """
    Read the TSV incrementally and collect a reproducible sample.

    Column 0 contains the incorrect sentence.
    Column 1 contains the corrected sentence.
    """
    random_generator = random.Random(RANDOM_SEED)

    examples: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        for row_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                continue

            incorrect = normalize_text(row[0])
            corrected = normalize_text(row[1])

            if not incorrect or not corrected:
                continue

            if incorrect == corrected:
                continue

            incorrect_length = len(incorrect.split())
            corrected_length = len(corrected.split())

            # Keep sequences manageable for an LSTM course project.
            if not 3 <= incorrect_length <= 80:
                continue

            if not 3 <= corrected_length <= 80:
                continue

            pair = (incorrect, corrected)

            if pair in seen_pairs:
                continue

            # Randomly select examples instead of taking only the
            # first 50,000 records from this shard.
            if random_generator.random() > 0.10:
                continue

            seen_pairs.add(pair)

            examples.append(
                {
                    "incorrect": incorrect,
                    "corrected": corrected,
                }
            )

            if len(examples) >= SAMPLE_SIZE:
                break

            if row_number % 500_000 == 0:
                print(
                    f"Read {row_number:,} rows; "
                    f"collected {len(examples):,} examples."
                )

    if len(examples) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(examples):,} usable examples were collected. "
            f"Expected {SAMPLE_SIZE:,}."
        )

    return pd.DataFrame(examples)


def split_examples(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle once with a fixed seed, then create held-out splits."""
    shuffled = dataframe.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    train = shuffled.iloc[:TRAIN_SIZE].copy()

    validation_start = TRAIN_SIZE
    validation_end = TRAIN_SIZE + VALIDATION_SIZE

    validation = shuffled.iloc[
        validation_start:validation_end
    ].copy()

    test = shuffled.iloc[
        validation_end:validation_end + TEST_SIZE
    ].copy()

    return train, validation, test


def save_split(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIRECTORY / filename

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved {len(dataframe):,} rows to {output_path}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {INPUT_FILE}"
        )

    print("Collecting 50,000 examples...")
    dataframe = collect_examples()

    train, validation, test = split_examples(dataframe)

    save_split(train, "train.csv")
    save_split(validation, "val.csv")
    save_split(test, "test.csv")

    print("\nFinal split sizes:")
    print(f"Training:   {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test:       {len(test):,}")


if __name__ == "__main__":
    main()