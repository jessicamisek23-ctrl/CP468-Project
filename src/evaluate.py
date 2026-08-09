from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.dataset import build_data_pipeline
from src.metrics import calculate_gleu, exact_match_accuracy
from src.seq2seq import build_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("models/best_model.pt")

OUTPUT_PATH = Path(
    "outputs/predictions/lstm_predictions.csv"
)

METRICS_PATH = Path(
    "outputs/metrics/lstm_metrics.json"
)


def detokenize(tokens: list[str]) -> str:
    """
    Convert tokenized model output into readable text.
    """
    if not tokens:
        return ""

    text = " ".join(tokens)

    # Common punctuation spacing.
    for punctuation in [".", ",", "!", "?", ";", ":"]:
        text = text.replace(
            f" {punctuation}",
            punctuation,
        )

    text = text.replace(" n't", "n't")
    text = text.replace(" 's", "'s")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    text = text.replace(" 'm", "'m")

    return text.strip()


@torch.no_grad()
def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/best_model.pt was not found. "
            "Train the model first."
        )

    pipeline = build_data_pipeline()

    vocabulary = pipeline["vocabulary"]
    test_loader = pipeline["test_loader"]

    model = build_model(
        vocabulary_size=len(vocabulary),
        pad_index=vocabulary.pad_index,
        sos_index=vocabulary.sos_index,
        eos_index=vocabulary.eos_index,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model.eval()

    source_sentences = []
    reference_sentences = []
    predictions = []

    start_time = time.perf_counter()

    for source, lengths, target in test_loader:
        source = source.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target = target.to(DEVICE)

        generated, _ = model.generate(
            source,
            lengths,
            max_length=80,
        )

        batch_size = source.shape[1]

        for index in range(batch_size):
            source_ids = source[:, index].tolist()
            target_ids = target[:, index].tolist()

            generated_ids = (
                generated[:, index].tolist()
            )

            source_tokens = vocabulary.decode(
                source_ids,
                skip_special_tokens=True,
                stop_at_eos=True,
            )

            reference_tokens = vocabulary.decode(
                target_ids,
                skip_special_tokens=True,
                stop_at_eos=True,
            )

            prediction_tokens = vocabulary.decode(
                generated_ids,
                skip_special_tokens=True,
                stop_at_eos=True,
            )

            source_sentences.append(
                detokenize(source_tokens)
            )

            reference_sentences.append(
                detokenize(reference_tokens)
            )

            predictions.append(
                detokenize(prediction_tokens)
            )

    inference_time = (
        time.perf_counter() - start_time
    )

    gleu = calculate_gleu(
        reference_sentences,
        predictions,
    )

    exact_match = exact_match_accuracy(
        reference_sentences,
        predictions,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        {
            "incorrect": source_sentences,
            "reference": reference_sentences,
            "lstm_output": predictions,
        }
    )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    results = {
        "test_examples": len(dataframe),
        "gleu": gleu,
        "exact_match_accuracy": exact_match,
        "inference_time_seconds": inference_time,
        "device": str(DEVICE),
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\nLSTM Test Results")
    print("-----------------")
    print(f"Examples: {len(dataframe):,}")
    print(f"GLEU: {gleu:.4f}")
    print(f"Exact match: {exact_match:.4f}")
    print(
        f"Inference time: {inference_time:.2f}s"
    )

    print(
        f"\nPredictions saved to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()