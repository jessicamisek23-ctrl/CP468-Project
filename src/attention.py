from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau-style additive attention over a bidirectional encoder's outputs.

    Given the decoder's current hidden state and all encoder outputs, scores
    each source position, masks out padding positions, and returns a
    weighted context vector.
    """

    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()

        # encoder_hidden_size is the per-direction size; the encoder is
        # bidirectional, so its outputs are of size encoder_hidden_size * 2.
        self.encoder_projection = nn.Linear(
            encoder_hidden_size * 2,
            attention_dim,
            bias=False,
        )

        self.decoder_projection = nn.Linear(
            decoder_hidden_size,
            attention_dim,
            bias=False,
        )

        self.energy_projection = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [batch, decoder_hidden_size]
                Top-layer decoder hidden state at the current timestep.
            encoder_outputs: [src_len, batch, encoder_hidden_size * 2]
            mask: [batch, src_len], True at padding positions (to be ignored).

        Returns:
            context: [batch, encoder_hidden_size * 2]
            attention_weights: [batch, src_len]
        """
        decoder_projected = self.decoder_projection(decoder_hidden).unsqueeze(0)
        # decoder_projected: [1, batch, attention_dim], broadcasts over src_len.

        encoder_projected = self.encoder_projection(encoder_outputs)
        # encoder_projected: [src_len, batch, attention_dim]

        energy = self.energy_projection(
            torch.tanh(encoder_projected + decoder_projected)
        )
        # energy: [src_len, batch, 1]

        energy = energy.squeeze(2).transpose(0, 1)
        # energy: [batch, src_len]

        energy = energy.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(energy, dim=1)
        # attention_weights: [batch, src_len]

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs.transpose(0, 1),
        ).squeeze(1)
        # context: [batch, encoder_hidden_size * 2]

        return context, attention_weights