# Dataset Setup

This project uses the **C4_200M Synthetic Grammatical Error Correction (GEC)** dataset.

The raw dataset is **not included** in this repository because it is several gigabytes in size. Instead, this repository provides scripts to reproduce the processed dataset used for all experiments.

---

# Dataset

**Task**

Grammatical Error Correction (GEC)

Input:

```text
Incorrect sentence
```

Output:

```text
Corrected sentence
```

Example:

```text
Incorrect:
She don't like apples.

Corrected:
She doesn't like apples.
```

---

# Download the Dataset

Download **one shard** of the C4_200M dataset.

The project was developed using the following shard:

```text
C4_200M.tsv-00004-of-00010
```

Place the downloaded file into:

```text
data/raw/
```

Your directory should look like:

```text
data/
│
├── raw/
│   └── C4_200M.tsv-00004-of-00010
│
└── processed/
```

---

# Dataset Format

The TSV file contains **no header**.

Each row contains two columns:

| Column | Description |
|--------|-------------|
| 0 | Incorrect sentence |
| 1 | Corrected sentence |

Example:

| Incorrect | Corrected |
|------------|-----------|
| She don't like apples. | She doesn't like apples. |

---

# Prepare the Dataset

From the repository root run:

```bash
python -m scripts.prepare_dataset
```

The preparation script will:

- read the raw TSV file;
- randomly sample **50,000** examples using a fixed seed;
- remove duplicate sentence pairs;
- normalize whitespace;
- create the training, validation, and test splits.

---

# Generated Files

After running the preparation script, the following files will be created:

```text
data/
│
├── processed/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── vocabulary.json
```

The generated files are ignored by Git and can always be recreated.

---

# Dataset Split

The processed dataset contains **50,000 sentence pairs**.

| Split | Examples |
|--------|---------:|
| Training | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |
| Total | 50,000 |

---

# Random Seed

To ensure reproducibility, all dataset sampling uses:

```text
468
```

Using the same seed produces the same dataset split.

---

# Verify the Dataset

After preparing the dataset, verify the preprocessing pipeline:

```bash
python -m scripts.test_preprocess
```

Then verify the DataLoader:

```bash
python -m scripts.test_data_pipeline
```

The output should display:

- vocabulary size;
- batch dimensions;
- decoded source sentence;
- decoded target sentence.

---

# Data Pipeline

The data preparation workflow is:

```text
Raw TSV
    │
    ▼
prepare_dataset.py
    │
    ▼
train.csv
val.csv
test.csv
    │
    ▼
preprocess.py
(cleaning + tokenization + vocabulary)
    │
    ▼
dataset.py
(PyTorch Dataset)
    │
    ▼
DataLoader
```

---

# Vocabulary

Vocabulary construction is performed automatically from the **training split only**.

Default settings:

| Parameter | Value |
|-----------|------:|
| Maximum vocabulary size | 30,000 |
| Minimum frequency | 2 |

Special tokens:

| Token | Index |
|--------|------:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<SOS>` | 2 |
| `<EOS>` | 3 |

---

# Generated CSV Format

Each generated CSV contains two columns:

| Column | Description |
|--------|-------------|
| incorrect | Source sentence |
| corrected | Target sentence |

Example:

| incorrect | corrected |
|------------|-----------|
| She don't like apples. | She doesn't like apples. |

---

# Notes

The raw C4_200M dataset is intentionally excluded from Git because of its size.

The processed dataset files (`train.csv`, `val.csv`, `test.csv`, and `vocabulary.json`) are also excluded because they can be regenerated at any time by running:

```bash
python -m scripts.prepare_dataset
```

This ensures that anyone cloning the repository can reproduce the exact dataset used in the experiments without storing large files in version control.