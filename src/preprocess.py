from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
]


def clean_text(text: str) -> str:
    """
    Clean spacing while preserving capitalization and punctuation.
    """
    text = str(text).strip()

    # Normalize quotation marks.
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Replace repeated whitespace with one space.
    text = re.sub(r"\s+", " ", text)

    return text


def tokenize(text: str) -> list[str]:
    """
    Split a sentence into words, contractions, numbers,
    and punctuation marks.

    Example:
        "She doesn't like apples."
        becomes:
        ["She", "doesn't", "like", "apples", "."]
    """
    cleaned_text = clean_text(text)

    return re.findall(
        r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]",
        cleaned_text,
    )


class Vocabulary:
    def __init__(
        self,
        minimum_frequency: int = 2,
        maximum_size: int | None = 30_000,
    ) -> None:
        if minimum_frequency < 1:
            raise ValueError(
                "minimum_frequency must be at least 1."
            )

        if maximum_size is not None and maximum_size < 4:
            raise ValueError(
                "maximum_size must be at least 4."
            )

        self.minimum_frequency = minimum_frequency
        self.maximum_size = maximum_size

        self.token_to_index: dict[str, int] = {
            token: index
            for index, token in enumerate(SPECIAL_TOKENS)
        }

        self.index_to_token: dict[int, str] = {
            index: token
            for token, index in self.token_to_index.items()
        }

    def __len__(self) -> int:
        return len(self.token_to_index)

    @property
    def pad_index(self) -> int:
        return self.token_to_index[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.token_to_index[UNK_TOKEN]

    @property
    def sos_index(self) -> int:
        return self.token_to_index[SOS_TOKEN]

    @property
    def eos_index(self) -> int:
        return self.token_to_index[EOS_TOKEN]

    def build(
        self,
        tokenized_sentences: Iterable[list[str]],
    ) -> None:
        """
        Build the vocabulary from tokenized training sentences.
        """
        counter: Counter[str] = Counter()

        for tokens in tokenized_sentences:
            counter.update(tokens)

        eligible_tokens = [
            (token, frequency)
            for token, frequency in counter.items()
            if frequency >= self.minimum_frequency
            and token not in self.token_to_index
        ]

        # Sort by frequency, then alphabetically for reproducibility.
        eligible_tokens.sort(
            key=lambda item: (-item[1], item[0])
        )

        if self.maximum_size is not None:
            available_positions = (
                self.maximum_size - len(SPECIAL_TOKENS)
            )

            eligible_tokens = eligible_tokens[
                :available_positions
            ]

        for token, _ in eligible_tokens:
            index = len(self.token_to_index)

            self.token_to_index[token] = index
            self.index_to_token[index] = token

    def encode(
        self,
        tokens: list[str],
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Convert tokens into vocabulary indices.
        """
        indices: list[int] = []

        if add_sos:
            indices.append(self.sos_index)

        indices.extend(
            self.token_to_index.get(
                token,
                self.unk_index,
            )
            for token in tokens
        )

        if add_eos:
            indices.append(self.eos_index)

        return indices

    def decode(
        self,
        indices: Iterable[int],
        skip_special_tokens: bool = False,
        stop_at_eos: bool = False,
    ) -> list[str]:
        """
        Convert vocabulary indices back into tokens.
        """
        tokens: list[str] = []

        for raw_index in indices:
            token = self.index_to_token.get(
                int(raw_index),
                UNK_TOKEN,
            )

            if stop_at_eos and token == EOS_TOKEN:
                break

            if (
                skip_special_tokens
                and token in SPECIAL_TOKENS
            ):
                continue

            tokens.append(token)

        return tokens

    def save(self, path: str | Path) -> None:
        """
        Save the vocabulary as JSON.
        """
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        contents = {
            "minimum_frequency": self.minimum_frequency,
            "maximum_size": self.maximum_size,
            "token_to_index": self.token_to_index,
        }

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                contents,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """
        Load a previously saved vocabulary.
        """
        with Path(path).open(
            "r",
            encoding="utf-8",
        ) as file:
            contents = json.load(file)

        vocabulary = cls(
            minimum_frequency=contents[
                "minimum_frequency"
            ],
            maximum_size=contents["maximum_size"],
        )

        vocabulary.token_to_index = {
            token: int(index)
            for token, index
            in contents["token_to_index"].items()
        }

        vocabulary.index_to_token = {
            index: token
            for token, index
            in vocabulary.token_to_index.items()
        }

        return vocabulary