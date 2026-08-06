from pathlib import Path
import csv
import sys


RAW_DIRECTORY = Path("data/raw")


def main() -> None:
    dataset_files = sorted(
        path
        for path in RAW_DIRECTORY.iterdir()
        if path.is_file() and ".tsv" in path.name.lower()
    )

    if not dataset_files:
        print("No TSV dataset file was found in data/raw.")
        sys.exit(1)

    path = dataset_files[0]

    print(f"Inspecting: {path}")
    print(f"File size: {path.stat().st_size / (1024 ** 3):.2f} GB")

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        for row_number, row in enumerate(reader, start=1):
            print(f"\nRow {row_number}")
            print(f"Number of columns: {len(row)}")

            for column_number, value in enumerate(row):
                preview = value[:300].replace("\n", " ")
                print(f"Column {column_number}: {preview}")

            if row_number == 5:
                break


if __name__ == "__main__":
    main()