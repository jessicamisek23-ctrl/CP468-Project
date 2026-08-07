from __future__ import annotations

import random

import torch
import torch.nn as nn

from src.attention import Attention
from src.decoder import Decoder
from src.encoder import Encoder


class Seq2Seq(nn.Module):
    """
    Ties the encoder, attention, and decoder together.

    Two entry points:
        forward()  - teacher-forced training pass, returns per-step logits.
        generate() - greedy decoding for evaluation / inference.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        pad_index: int,
        sos_index: int,
        eos_index: int,
    ) -> None:
        super().__init__()

        if encoder.num_layers != decoder.num_layers:
            raise ValueError(
                "encoder.num_layers must match decoder.num_layers, since "
                "the encoder's final state initializes the decoder."
            )

        self.encoder = encoder
        self.decoder = decoder

        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def create_mask(self, source: torch.Tensor) -> torch.Tensor:
        """
        source: [src_len, batch] -> mask: [batch, src_len]
        True at positions that are padding (to be excluded from attention).
        """
        return (source == self.pad_index).transpose(0, 1)

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            source: [src_len, batch]
            source_lengths: [batch], sorted descending.
            target: [tgt_len, batch], target[0] is the <SOS> token for every
                sequence in the batch (as produced by GrammarCorrectionDataset).
            teacher_forcing_ratio: probability of feeding the ground-truth
                previous token instead of the model's own prediction.

        Returns:
            outputs: [tgt_len, batch, vocabulary_size]
                outputs[0] is left as zeros (there is nothing to predict for
                the <SOS> slot). Compute loss on outputs[1:] vs target[1:].
        """
        batch_size = source.size(1)
        target_length = target.size(0)
        vocabulary_size = self.decoder.vocabulary_size

        outputs = torch.zeros(
            target_length,
            batch_size,
            vocabulary_size,
            device=self.device,
        )

        encoder_outputs, hidden, cell = self.encoder(source, source_lengths)
        mask = self.create_mask(source)

        input_token = target[0]

        for t in range(1, target_length):
            logits, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                mask,
            )

            outputs[t] = logits

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)

            input_token = target[t] if use_teacher_forcing else top1

        return outputs

    @torch.no_grad()
    def generate(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        max_length: int = 80,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding, for validation/test evaluation and qualitative
        error analysis (no ground-truth target is used).

        Args:
            source: [src_len, batch]
            source_lengths: [batch], sorted descending.
            max_length: maximum number of tokens to generate.

        Returns:
            predictions: [output_len, batch] generated token indices.
                Sequences that finish early are padded with pad_index.
            attention_weights: [output_len, batch, src_len]
                Attention distributions at each generated step, useful for
                qualitative analysis / attention visualization.
        """
        was_training = self.training
        self.eval()

        batch_size = source.size(1)
        src_len = source.size(0)

        encoder_outputs, hidden, cell = self.encoder(source, source_lengths)
        mask = self.create_mask(source)

        input_token = torch.full(
            (batch_size,),
            self.sos_index,
            dtype=torch.long,
            device=self.device,
        )

        predictions = torch.full(
            (max_length, batch_size),
            self.pad_index,
            dtype=torch.long,
            device=self.device,
        )

        attention_weights_all = torch.zeros(
            max_length,
            batch_size,
            src_len,
            device=self.device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        steps_taken = 0

        for t in range(max_length):
            logits, hidden, cell, attention_weights = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                mask,
            )

            top1 = logits.argmax(dim=1)
            top1 = torch.where(
                finished,
                torch.full_like(top1, self.pad_index),
                top1,
            )

            predictions[t] = top1
            attention_weights_all[t] = attention_weights

            finished = finished | (top1 == self.eos_index)
            input_token = top1
            steps_taken = t + 1

            if finished.all():
                break

        if was_training:
            self.train()

        return predictions[:steps_taken], attention_weights_all[:steps_taken]


def build_model(
    vocabulary_size: int,
    pad_index: int,
    sos_index: int,
    eos_index: int,
    embedding_dim: int = 256,
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
    attention_dim: int = 256,
) -> Seq2Seq:
    """
    Convenience factory that wires up Encoder + Attention + Decoder with
    consistent dimensions, matching the minimum architecture required by the
    project spec: embedding -> bidirectional LSTM encoder -> attention ->
    LSTM decoder -> output projection.
    """
    encoder = Encoder(
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_index=pad_index,
    )

    attention = Attention(
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        attention_dim=attention_dim,
    )

    decoder = Decoder(
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        encoder_hidden_size=hidden_size,
        decoder_hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_index=pad_index,
        attention=attention,
    )

    return Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        pad_index=pad_index,
        sos_index=sos_index,
        eos_index=eos_index,
    )