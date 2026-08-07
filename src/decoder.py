from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import Attention


class Decoder(nn.Module):
    """
    Attention-based LSTM decoder. Operates one timestep at a time so that
    Seq2Seq can drive it with teacher forcing during training and greedy
    (or beam) search during inference.

    Note: `num_layers` here must match the encoder's `num_layers`, since it
    receives the encoder's projected hidden/cell state directly.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 256,
        encoder_hidden_size: int = 512,
        decoder_hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_index: int = 0,
        attention: Attention | None = None,
    ) -> None:
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_size = decoder_hidden_size
        self.num_layers = num_layers

        self.attention = attention or Attention(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
        )

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=pad_index,
        )

        self.dropout = nn.Dropout(dropout)

        # Input feeding: the previous-token embedding is concatenated with
        # the attention context vector at every decoder step.
        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_hidden_size * 2,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_projection = nn.Linear(
            decoder_hidden_size + encoder_hidden_size * 2 + embedding_dim,
            vocabulary_size,
        )

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_token: [batch] token indices fed in at this step (either
                the ground-truth previous token during teacher forcing, or
                the model's own previous prediction).
            hidden: [num_layers, batch, decoder_hidden_size]
            cell: [num_layers, batch, decoder_hidden_size]
            encoder_outputs: [src_len, batch, encoder_hidden_size * 2]
            mask: [batch, src_len], True at padding positions.

        Returns:
            logits: [batch, vocabulary_size]
            hidden: [num_layers, batch, decoder_hidden_size]
            cell: [num_layers, batch, decoder_hidden_size]
            attention_weights: [batch, src_len]
        """
        input_token = input_token.unsqueeze(0)
        # input_token: [1, batch]

        embedded = self.dropout(self.embedding(input_token))
        # embedded: [1, batch, embedding_dim]

        context, attention_weights = self.attention(
            hidden[-1],
            encoder_outputs,
            mask,
        )
        context = context.unsqueeze(0)
        # context: [1, batch, encoder_hidden_size * 2]

        lstm_input = torch.cat([embedded, context], dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: [1, batch, decoder_hidden_size]

        output = output.squeeze(0)
        embedded = embedded.squeeze(0)
        context = context.squeeze(0)

        logits = self.output_projection(
            torch.cat([output, context, embedded], dim=1)
        )

        return logits, hidden, cell, attention_weights