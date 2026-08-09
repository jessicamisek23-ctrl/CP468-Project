from pathlib import Path

import pandas as pd


LSTM_PATH = Path(
    "outputs/predictions/lstm_predictions.csv"
)

GEMINI_PATH = Path(
    "outputs/predictions/"
    "gemini_few_shot_predictions.csv"
)

OUTPUT_PATH = Path(
    "outputs/predictions/"
    "qualitative_candidates.csv"
)


def main() -> None:
    lstm = pd.read_csv(LSTM_PATH)
    gemini = pd.read_csv(GEMINI_PATH)

    if len(lstm) != len(gemini):
        raise ValueError(
            "LSTM and Gemini prediction files "
            "must contain the same test examples."
        )

    comparison = pd.DataFrame(
        {
            "incorrect": lstm["incorrect"],
            "reference": lstm["reference"],
            "lstm_output": lstm["lstm_output"],
            "gemini_output": (
                gemini["gemini_output"]
            ),
        }
    )

    comparison["lstm_exact"] = (
        comparison["lstm_output"]
        == comparison["reference"]
    )

    comparison["gemini_exact"] = (
        comparison["gemini_output"]
        == comparison["reference"]
    )

    comparison["source_length"] = (
        comparison["incorrect"]
        .str.split()
        .str.len()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(comparison):,} "
        f"candidate examples to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()