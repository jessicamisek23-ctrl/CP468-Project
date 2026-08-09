from __future__ import annotations

from nltk.translate.gleu_score import corpus_gleu

from src.preprocess import tokenize


def calculate_gleu(
    references: list[str],
    predictions: list[str],
) -> float:
    if len(references) != len(predictions):
        raise ValueError(
            "References and predictions must have "
            "the same length."
        )

    tokenized_references = [
        [tokenize(reference)]
        for reference in references
    ]

    tokenized_predictions = [
        tokenize(prediction)
        for prediction in predictions
    ]

    return float(
        corpus_gleu(
            tokenized_references,
            tokenized_predictions,
        )
    )


def exact_match_accuracy(
    references: list[str],
    predictions: list[str],
) -> float:
    if not references:
        return 0.0

    matches = sum(
        reference.strip() == prediction.strip()
        for reference, prediction
        in zip(references, predictions)
    )

    return matches / len(references)