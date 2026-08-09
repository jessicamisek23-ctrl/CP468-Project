from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

from src.metrics import (
    calculate_gleu,
    exact_match_accuracy,
)


TEST_PATH = Path("data/processed/test.csv")

ZERO_SHOT_PATH = Path(
    "prompts/zero_shot.txt"
)

FEW_SHOT_PATH = Path(
    "prompts/few_shot.txt"
)

OUTPUT_DIRECTORY = Path(
    "outputs/predictions"
)

METRICS_DIRECTORY = Path(
    "outputs/metrics"
)

# Lets you change models without editing source code.
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)


def load_prompt(path: Path) -> str:
    return path.read_text(
        encoding="utf-8"
    )


def correct_sentence(
    client: genai.Client,
    sentence: str,
    prompt_template: str,
) -> tuple[str, int, int]:
    prompt = prompt_template.format(
        sentence=sentence
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=128,
        ),
    )

    text = response.text.strip()

    input_tokens = 0
    output_tokens = 0

    if response.usage_metadata is not None:
        input_tokens = (
            response.usage_metadata.prompt_token_count
            or 0
        )

        output_tokens = (
            response.usage_metadata.candidates_token_count
            or 0
        )

    return text, input_tokens, output_tokens


def run_prompt_setting(
    test: pd.DataFrame,
    prompt_name: str,
    prompt_template: str,
) -> None:
    client = genai.Client()

    output_path = (
        OUTPUT_DIRECTORY
        / f"gemini_{prompt_name}_predictions.csv"
    )

    # Resume an interrupted run.
    if output_path.exists():
        completed = pd.read_csv(output_path)
        completed_count = len(completed)

        outputs = completed[
            "gemini_output"
        ].tolist()

        input_tokens = int(
            completed[
                "input_tokens"
            ].sum()
        )

        output_tokens = int(
            completed[
                "output_tokens"
            ].sum()
        )

        print(
            f"Resuming {prompt_name} from "
            f"example {completed_count:,}."
        )

    else:
        completed_count = 0
        outputs = []
        input_tokens = 0
        output_tokens = 0

    start_time = time.perf_counter()

    for index in range(
        completed_count,
        len(test),
    ):
        sentence = test.iloc[index][
            "incorrect"
        ]

        corrected, used_input, used_output = (
            correct_sentence(
                client,
                sentence,
                prompt_template,
            )
        )

        outputs.append(corrected)

        input_tokens += used_input
        output_tokens += used_output

        # Save frequently so an interrupted API run
        # does not lose all progress.
        if (
            (index + 1) % 25 == 0
            or index == len(test) - 1
        ):
            partial = test.iloc[
                : len(outputs)
            ].copy()

            partial["gemini_output"] = outputs

            # Store cumulative totals in the last row
            # through per-row approximations is awkward,
            # so write token metadata columns using 0 and
            # the current totals separately below.
            partial["input_tokens"] = 0
            partial["output_tokens"] = 0

            partial.to_csv(
                output_path,
                index=False,
            )

            print(
                f"{prompt_name}: "
                f"{len(outputs):,}/{len(test):,}"
            )

    elapsed = time.perf_counter() - start_time

    dataframe = test.copy()
    dataframe["gemini_output"] = outputs

    dataframe.to_csv(
        output_path,
        index=False,
    )

    gleu = calculate_gleu(
        dataframe["corrected"].tolist(),
        dataframe["gemini_output"].tolist(),
    )

    exact_match = exact_match_accuracy(
        dataframe["corrected"].tolist(),
        dataframe["gemini_output"].tolist(),
    )

    metrics = {
        "model": MODEL_NAME,
        "prompt_setting": prompt_name,
        "examples": len(dataframe),
        "gleu": gleu,
        "exact_match_accuracy": exact_match,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "runtime_seconds": elapsed,
    }

    metrics_path = (
        METRICS_DIRECTORY
        / f"gemini_{prompt_name}_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"\n{prompt_name} complete:"
    )
    print(f"GLEU: {gleu:.4f}")
    print(
        f"Exact match: {exact_match:.4f}"
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed test set not found."
        )

    test = pd.read_csv(TEST_PATH)

    zero_shot_prompt = load_prompt(
        ZERO_SHOT_PATH
    )

    few_shot_prompt = load_prompt(
        FEW_SHOT_PATH
    )

    run_prompt_setting(
        test,
        "zero_shot",
        zero_shot_prompt,
    )

    run_prompt_setting(
        test,
        "few_shot",
        few_shot_prompt,
    )


if __name__ == "__main__":
    main()