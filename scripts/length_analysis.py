from pathlib import Path

import pandas as pd

from src.metrics import calculate_gleu


INPUT_PATH = Path(
    "outputs/predictions/"
    "qualitative_candidates.csv"
)


def main() -> None:
    dataframe = pd.read_csv(INPUT_PATH)

    dataframe["length_bucket"] = pd.cut(
        dataframe["source_length"],
        bins=[0, 10, 20, 40, 80],
        labels=[
            "1-10",
            "11-20",
            "21-40",
            "41-80",
        ],
    )

    results = []

    for bucket, group in dataframe.groupby(
        "length_bucket",
        observed=True,
    ):
        lstm_gleu = calculate_gleu(
            group["reference"].tolist(),
            group["lstm_output"].tolist(),
        )

        gemini_gleu = calculate_gleu(
            group["reference"].tolist(),
            group["gemini_output"].tolist(),
        )

        results.append(
            {
                "length_bucket": str(bucket),
                "examples": len(group),
                "lstm_gleu": lstm_gleu,
                "gemini_gleu": gemini_gleu,
            }
        )

    output = pd.DataFrame(results)

    output.to_csv(
        "outputs/metrics/"
        "length_analysis.csv",
        index=False,
    )

    print(output.to_string(index=False))


if __name__ == "__main__":
    main()