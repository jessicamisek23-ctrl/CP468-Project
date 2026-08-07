from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Embedding layer -> bidirectional multi-layer LSTM encoder.

    Expects batch_first=False tensors, matching src/dataset.py:
        source: [src_len, batch]
        source_lengths: [batch] (sorted descending, as produced by collate_batch)

    The encoder's final forward/backward hidden and cell states are
    concatenated and projected down to `hidden_size` so they can be used
    directly as the initial state of a (unidirectional) decoder LSTM.
    Note: `num_layers` here must match the decoder's `num_layers`, since the
    projected state is passed straight into the decoder LSTM.
    """

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
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Concatenated forward+backward final states -> decoder hidden size.
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            source: [src_len, batch] token indices.
            source_lengths: [batch] true (unpadded) lengths, sorted descending.

        Returns:
            encoder_outputs: [src_len, batch, hidden_size * 2]
                Per-timestep outputs, used by the attention mechanism.
                Padded positions contain zeros and are masked out downstream.
            hidden: [num_layers, batch, hidden_size]
                Initial decoder hidden state.
            cell: [num_layers, batch, hidden_size]
                Initial decoder cell state.
        """
        embedded = self.dropout(self.embedding(source))

        packed = pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            enforce_sorted=True,
        )

        packed_outputs, (hidden, cell) = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(packed_outputs)

        hidden = self._combine_directions(hidden, self.hidden_projection)
        cell = self._combine_directions(cell, self.cell_projection)

        return encoder_outputs, hidden, cell

    def _combine_directions(
        self,
        state: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        """
        Reshape [num_layers * 2, batch, hidden_size] (forward/backward
        interleaved per layer, as returned by nn.LSTM) into
        [num_layers, batch, hidden_size] by concatenating each layer's
        forward and backward states and projecting them down.
        """
        num_directions = 2
        num_layers = state.size(0) // num_directions
        batch_size = state.size(1)

        state = state.view(
            num_layers,
            num_directions,
            batch_size,
            self.hidden_size,
        )

        combined = torch.cat([state[:, 0], state[:, 1]], dim=2)

        return torch.tanh(projection(combined))