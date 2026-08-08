from openai import OpenAI
import pandas as pd

client = OpenAI()


def correct_sentence(sentence):

    prompt = f"""
Correct the grammar.

Only return the corrected sentence.

Sentence:
{sentence}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    return response.output_text.strip()


# Load test set
test = pd.read_csv(
    "data/processed/test.csv"
)

outputs = []

# Run LLM on every test sentence
for sentence in test["incorrect"]:

    corrected = correct_sentence(sentence)

    outputs.append(corrected)

# Save predictions
test["llm_output"] = outputs

test.to_csv(
    "llm_predictions.csv",
    index=False,
)

print("Finished.")