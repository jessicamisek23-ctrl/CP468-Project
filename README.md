# CP468 Course Project
## Sequence-to-Sequence Modeling: LSTM vs. LLM

### Grammatical Error Correction using an Attention-Based LSTM and Gemini

---

## 1. Project Overview

This project implements and evaluates a classical sequence-to-sequence (Seq2Seq) model for **Grammatical Error Correction (GEC)**.

The task is formulated as:

```text
Incorrect sentence → Corrected sentence
```

Example:

```text
Input:
She don't like apples.

Target:
She doesn't like apples.
```

The project compares a task-specific LSTM Seq2Seq model trained from scratch against a modern Large Language Model (LLM) baseline.

The systems compared are:

1. **LSTM Seq2Seq with attention**
2. **LSTM Seq2Seq without attention** — ablation experiment
3. **Gemini zero-shot**
4. **Gemini few-shot**

The purpose of the project is not to outperform the LLM, but to compare model quality and analyze differences in:

- generation performance;
- model capacity;
- training requirements;
- inference cost;
- latency;
- controllability;
- failure modes;
- engineering trade-offs.

---

# 2. Task

The selected task is:

## Grammatical Error Correction

The model receives an English sentence containing one or more grammatical errors and attempts to produce the corrected version of the sentence.

```text
Source sequence:
Incorrect English sentence

Target sequence:
Corrected English sentence
```

Both the source and target are English, so the project uses a **shared vocabulary** for the encoder and decoder.

---

# 3. Dataset

## C4_200M Synthetic Grammatical Error Correction Dataset

This project uses the **C4_200M Synthetic Grammatical Error Correction dataset**.

Each TSV record contains:

| Column | Description |
|---|---|
| 0 | Incorrect / corrupted sentence |
| 1 | Corrected sentence |

The raw dataset is extremely large, so this project uses a reproducibly sampled subset from one shard:

```text
C4_200M.tsv-00004-of-00010
```

The raw dataset file is **not stored in this GitHub repository** because of its size.

See:

```text
data/README.md
```

for dataset download and preparation instructions.

---

# 4. Dataset Split

A fixed subset of **50,000 sentence pairs** is used.

| Split | Examples |
|---|---:|
| Training | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |
| **Total** | **50,000** |

Random seed:

```text
468
```

The validation and test sets are held out from model training.

The vocabulary is constructed using **training data only** to prevent information leakage.

---

# 5. Preprocessing

The preprocessing pipeline is implemented in:

```text
src/preprocess.py
src/dataset.py
```

The pipeline performs:

- whitespace normalization;
- quotation-mark normalization;
- tokenization;
- vocabulary construction;
- unknown-token handling;
- sequence truncation;
- EOS/SOS token insertion;
- dynamic batch padding;
- sequence-length tracking;
- PyTorch Dataset creation;
- PyTorch DataLoader creation.

Capitalization and punctuation are deliberately preserved because correcting capitalization and punctuation is part of the grammatical error correction task.

---

## Tokenization Example

Input:

```text
She doesn't like apples.
```

Tokens:

```text
["She", "doesn't", "like", "apples", "."]
```

---

# 6. Vocabulary

The project uses a single shared vocabulary for both the encoder and decoder.

Default vocabulary settings:

| Setting | Value |
|---|---:|
| Maximum vocabulary size | 30,000 |
| Minimum token frequency | 2 |

Special tokens:

| Token | Index |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<SOS>` | 2 |
| `<EOS>` | 3 |

Source sequences are encoded as:

```text
sentence + <EOS>
```

Target sequences are encoded as:

```text
<SOS> + corrected sentence + <EOS>
```

The maximum sequence length is:

```text
80 tokens
```

---

# 7. DataLoader Interface

Each DataLoader batch returns:

```python
source, source_lengths, target
```

Tensor shapes are:

```text
source:
[source_length, batch_size]

source_lengths:
[batch_size]

target:
[target_length, batch_size]
```

Default batch size:

```text
32
```

The source sequences are sorted by descending length so that PyTorch packed sequences can be used efficiently by the encoder.

---

# 8. Model Architecture

The main model uses the following architecture:

```text
Input Sentence
      │
      ▼
Token Embedding
      │
      ▼
Bidirectional LSTM Encoder
      │
      ▼
Bahdanau Additive Attention
      │
      ▼
LSTM Decoder
      │
      ▼
Output Projection
      │
      ▼
Corrected Sentence
```

---

## Encoder

The encoder uses:

- token embeddings;
- dropout;
- a multi-layer bidirectional LSTM;
- packed padded sequences;
- linear projection of forward/backward final states.

Default encoder settings:

| Setting | Value |
|---|---:|
| Embedding dimension | 256 |
| Hidden size | 512 |
| LSTM layers | 2 |
| Bidirectional | Yes |
| Dropout | 0.3 |

---

## Attention

The main model uses **Bahdanau-style additive attention**.

At each decoder timestep, attention scores are computed over all encoder outputs.

Padding positions are masked before the attention softmax so that the model does not attend to `<PAD>` tokens.

Default attention dimension:

```text
256
```

---

## Decoder

The decoder uses:

- token embedding;
- attention context;
- LSTM recurrence;
- output projection to vocabulary logits.

At each timestep, the previous token embedding is concatenated with the attention context before being passed through the decoder LSTM.

---

# 9. Teacher Forcing

During training, the Seq2Seq model uses teacher forcing.

Default teacher forcing ratio:

```text
0.5
```

This means that at each decoder timestep there is a 50% probability of using the correct previous target token instead of the model's own previous prediction.

During evaluation and generation, teacher forcing is disabled.

---

# 10. Inference

The main model provides greedy autoregressive generation.

Generation begins with:

```text
<SOS>
```

The decoder repeatedly predicts the next token until:

```text
<EOS>
```

is generated or the maximum generation length is reached.

---

# 11. Ablation Experiment

The project includes a model without attention:

```text
src/no_attention.py
```

This model is used to evaluate the effect of the attention mechanism.

The comparison is:

```text
LSTM + Attention
       vs.
LSTM without Attention
```

The no-attention model uses the same general training configuration and dataset so that the comparison is meaningful.

---

# 12. LLM Baseline

The project uses **Google Gemini** as the modern LLM baseline.

Two prompt settings are evaluated:

### Zero-shot

The LLM receives only task instructions and the sentence to correct.

Prompt:

```text
prompts/zero_shot.txt
```

### Few-shot

The LLM receives task instructions plus several example grammatical corrections before receiving the test sentence.

Prompt:

```text
prompts/few_shot.txt
```

The LLM is evaluated using the **same held-out test examples** as the LSTM model.

The exact prompt text used in the experiments is stored in the repository for reproducibility.

---

# 13. Evaluation

All models are evaluated on the same held-out test set.

The primary automatic metric is:

```text
GLEU
```

Exact-match accuracy is also reported as a supplementary metric.

The systems evaluated are:

| Model | Evaluation |
|---|---|
| LSTM + Attention | GLEU + Exact Match |
| LSTM without Attention | GLEU + Exact Match |
| Gemini Zero-Shot | GLEU + Exact Match |
| Gemini Few-Shot | GLEU + Exact Match |

Additional analysis includes:

- qualitative error analysis;
- sentence-length analysis;
- failure-mode comparison;
- inference/runtime comparison.

At least 10 test examples are selected for detailed qualitative comparison.

Each qualitative example contains:

```text
Incorrect sentence
Reference correction
LSTM output
Gemini output
```

---

# 14. Repository Structure

```text
CP468-Project/
│
├── data/
│   ├── README.md
│   │
│   ├── raw/
│   │   └── C4_200M.tsv-00004-of-00010
│   │       [not committed to Git]
│   │
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── vocabulary.json
│       [generated locally]
│
├── models/
│   ├── best_model.pt
│   └── best_no_attention_model.pt
│   [generated locally]
│
├── outputs/
│   ├── metrics/
│   └── predictions/
│
├── prompts/
│   ├── zero_shot.txt
│   └── few_shot.txt
│
├── scripts/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   ├── test_preprocess.py
│   ├── test_data_pipeline.py
│   ├── build_qualitative_examples.py
│   └── length_analysis.py
│
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── dataset.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── evaluate.py
│   ├── llm_baseline.py
│   ├── metrics.py
│   ├── no_attention.py
│   ├── preprocess.py
│   ├── seq2seq.py
│   ├── train.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 15. Installation

## Step 1 — Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd CP468-Project
```

---

## Step 2 — Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
```

Then:

```bash
python -m pip install -r requirements.txt
```

---

# 16. Dataset Setup

Download one C4_200M TSV shard as described in:

```text
data/README.md
```

Place the file at:

```text
data/raw/C4_200M.tsv-00004-of-00010
```

The file contains no header.

```text
Column 0 = Incorrect sentence
Column 1 = Corrected sentence
```

---

# 17. Prepare the Dataset

From the repository root:

```bash
python -m scripts.prepare_dataset
```

This generates:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

with:

```text
Training:   40,000
Validation: 5,000
Test:       5,000
```

using random seed:

```text
468
```

---

# 18. Test Preprocessing

Run:

```bash
python -m scripts.test_preprocess
```

This verifies:

- text cleaning;
- tokenization;
- special tokens;
- vocabulary encoding;
- vocabulary decoding.

Expected special-token indices:

```text
PAD = 0
UNK = 1
SOS = 2
EOS = 3
```

---

# 19. Test the Data Pipeline

Run:

```bash
python -m scripts.test_data_pipeline
```

The script should print:

- source batch shape;
- source length shape;
- target batch shape;
- vocabulary size;
- token indices;
- a decoded incorrect sentence;
- the corresponding corrected sentence.

Expected DataLoader interface:

```python
source, source_lengths, target
```

---

# 20. Train the Main LSTM + Attention Model

Before running the full experiment, it is recommended to temporarily use:

```python
EPOCHS = 1
```

in `src/train.py` to verify that the entire training pipeline works.

Run:

```bash
python -m src.train
```

After the one-epoch integration test succeeds, restore:

```python
EPOCHS = 20
```

and run:

```bash
python -m src.train
```

The best model is saved to:

```text
models/best_model.pt
```

Training metadata is saved under:

```text
outputs/metrics/
```

The training script reports:

- training loss;
- validation loss;
- model parameter count;
- training time;
- hardware/device used.

---

# 21. Evaluate the Main LSTM

After training:

```bash
python -m src.evaluate
```

This loads:

```text
models/best_model.pt
```

and evaluates it on the held-out test set.

Predictions are saved to:

```text
outputs/predictions/lstm_predictions.csv
```

Metrics are saved to:

```text
outputs/metrics/lstm_metrics.json
```

---

# 22. Train the No-Attention Ablation

Run:

```bash
python -m src.no_attention
```

The best no-attention model is saved to:

```text
models/best_no_attention_model.pt
```

The results can then be compared against the attention-based model to measure the effect of attention.

---

# 23. Gemini API Setup

The Gemini baseline requires an API key.

Do **not** place API keys directly in source code or commit them to GitHub.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

### macOS / Linux

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

---

# 24. Run the Gemini Baseline

Run:

```bash
python -m src.llm_baseline
```

The script runs:

```text
Zero-shot evaluation
Few-shot evaluation
```

and saves the resulting predictions and metrics under:

```text
outputs/predictions/
outputs/metrics/
```

API token usage should be recorded so that the approximate Gemini API cost can be reported in the final report.

---

# 25. Build Qualitative Comparison Data

After both the LSTM and Gemini experiments have been completed:

```bash
python -m scripts.build_qualitative_examples
```

This combines the outputs into a comparison file containing:

```text
Incorrect
Reference
LSTM output
Gemini output
```

The file can be used to select at least 10 examples for the qualitative error analysis in the final report.

---

# 26. Run Sentence-Length Analysis

Run:

```bash
python -m scripts.length_analysis
```

This evaluates model performance across different source-length groups.

Example buckets:

```text
1–10 tokens
11–20 tokens
21–40 tokens
41–80 tokens
```

This helps determine whether the performance gap between the LSTM and Gemini changes as sequence length increases.

---

# 27. Recommended Full Reproduction Order

From a clean repository, run the project in the following order:

```bash
python -m scripts.prepare_dataset
```

```bash
python -m scripts.test_preprocess
```

```bash
python -m scripts.test_data_pipeline
```

```bash
python -m src.train
```

```bash
python -m src.evaluate
```

```bash
python -m src.no_attention
```

```bash
python -m src.llm_baseline
```

```bash
python -m scripts.build_qualitative_examples
```

```bash
python -m scripts.length_analysis
```

---

# 28. Main Experimental Settings

Unless otherwise stated in the final report, the main LSTM configuration is:

| Hyperparameter | Value |
|---|---:|
| Random seed | 468 |
| Training examples | 40,000 |
| Validation examples | 5,000 |
| Test examples | 5,000 |
| Vocabulary size | 30,000 |
| Minimum token frequency | 2 |
| Maximum sequence length | 80 |
| Batch size | 32 |
| Embedding dimension | 256 |
| Encoder hidden size | 512 |
| Decoder hidden size | 512 |
| LSTM layers | 2 |
| Encoder | Bidirectional |
| Dropout | 0.3 |
| Attention | Bahdanau additive |
| Attention dimension | 256 |
| Teacher forcing | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Gradient clipping | 1.0 |
| Epochs | 20 |
| Loss | Cross-Entropy |
| Padding ignored in loss | Yes |

---

# 29. Reproducibility

Randomness is controlled using:

```text
Seed = 468
```

The seed is applied to:

- Python random;
- NumPy;
- PyTorch;
- CUDA, when available;
- DataLoader shuffling.

The repository contains:

```text
requirements.txt
```

with the Python package versions needed to reproduce the experiments.

The raw dataset is not committed due to size, but the repository provides:

- the expected dataset location;
- preparation instructions;
- preprocessing scripts;
- fixed dataset sampling rules;
- fixed train/validation/test split sizes;
- a fixed random seed.

---

# 30. Expected Outputs

## Models

```text
models/best_model.pt
models/best_no_attention_model.pt
```

## Predictions

```text
outputs/predictions/lstm_predictions.csv
outputs/predictions/gemini_zero_shot_predictions.csv
outputs/predictions/gemini_few_shot_predictions.csv
outputs/predictions/qualitative_candidates.csv
```

## Metrics

```text
outputs/metrics/training_history.json
outputs/metrics/lstm_metrics.json
outputs/metrics/gemini_zero_shot_metrics.json
outputs/metrics/gemini_few_shot_metrics.json
outputs/metrics/length_analysis.csv
```

Exact filenames may vary slightly depending on the final implementation.

---

# 31. Final Comparison

The final report compares:

| System | GLEU | Exact Match | Runtime | Cost / Compute |
|---|---:|---:|---:|---|
| LSTM + Attention | TBD | TBD | TBD | Local compute |
| LSTM without Attention | TBD | TBD | TBD | Local compute |
| Gemini Zero-Shot | TBD | TBD | TBD | API cost |
| Gemini Few-Shot | TBD | TBD | TBD | API cost |

Results should be filled in after the final experiments are completed.

---

# 32. Qualitative Error Analysis

At least 10 representative test examples are analyzed.

Potential error categories include:

- subject–verb agreement;
- verb tense;
- articles and determiners;
- punctuation;
- capitalization;
- spelling;
- word omission;
- unnecessary word insertion;
- repetition;
- out-of-vocabulary failures;
- incorrect semantic changes;
- overcorrection;
- LLM hallucination.

Examples should be chosen to demonstrate different model behaviors rather than only successful cases.

---

# 33. Engineering Trade-Offs

The report discusses trade-offs between a task-specific LSTM and a general-purpose LLM, including:

- model size;
- training compute;
- inference latency;
- API cost;
- privacy;
- offline deployment;
- controllability;
- specialized-domain performance;
- data requirements;
- generalization.

---

# 34. Limitations

Important limitations considered include:

- synthetic nature of the C4_200M grammatical errors;
- automatic metric limitations;
- vocabulary-based out-of-vocabulary behavior;
- limited LSTM training data compared with LLM pretraining;
- possible LLM data contamination;
- compute limitations;
- differences in model scale;
- fairness of comparing a model trained from scratch with a pretrained foundation model.

---

# 35. Project Report

The final submission includes a system report covering:

- dataset;
- system design;
- experimental settings;
- quantitative results;
- qualitative error analysis;
- limitations;
- discussion;
- team contributions;
- AI-use disclosure.

Final report:

```text
ADD REPORT LINK HERE
```

---

# 36. Demo Video

The project includes an approximately 8-minute demonstration video.

Demo link:

```text
ADD DEMO VIDEO LINK HERE
```

---

# 37. Team Members and Contributions

| Team Member | Contribution |
|---|---|
| Member 1 | Model architecture / Encoder / Attention / Decoder |
| Member 2 | Dataset / Preprocessing / Vocabulary / DataLoaders |
| Member 3 | Training / Evaluation / LLM baseline |
| Member 4 | Experiments / Report |
| Member 5 | Analysis / Documentation / Demo |

Replace the placeholders above with the team's actual names and final contributions.

A detailed contribution statement is included in the report appendix.

---

# 38. AI Use Disclosure

Any use of generative AI tools during development, debugging, documentation, or report preparation is disclosed in the project report according to course requirements.

The final report contains the complete AI-use disclosure.

---

# 39. References

Include the final dataset and research citations here.

Suggested categories:

1. C4_200M Synthetic Grammatical Error Correction dataset.
2. Seq2Seq / encoder-decoder literature.
3. Bahdanau attention.
4. LSTM literature.
5. GLEU evaluation metric.
6. Gemini documentation/model information.

Full formatted references are provided in the project report.

---

# 40. License

This repository contains course-project code.

Dataset use and redistribution are subject to the terms and licensing of the original C4/C4_200M data sources.

The large raw dataset files and generated model checkpoints are intentionally excluded from Git.