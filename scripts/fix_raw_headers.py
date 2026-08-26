"""
Fix delimiter issues and restore lost header rows in data/raw/*.csv.

Rules applied (per file, based on inspection of the raw bytes):
- tamil_offensive_full*.csv        -> tab-delimited "offensive" files
- Abusive_ Tamil Text (*)          -> comma-delimited files
- tamil_abusive_triplets_*         -> already has a header/comma delimiter, untouched

The *_dev/_test/_train tab files also carry a stray trailing tab (an empty
3rd column) on every row; that artifact is dropped when the header is
restored.

Running this script overwrites the raw files in place with a header row
and a single, consistent delimiter.
"""

from __future__ import annotations

import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def restore_header(
    filename: str,
    delimiter: str,
    header: list[str],
    strip_trailing_empty_field: bool = False,
) -> None:
    path = RAW_DIR / filename
    with path.open("r", encoding="utf-8", newline="") as f:
        first_line = f.readline()

    # Idempotency: skip files that already have the target header.
    existing_first_fields = first_line.rstrip("\r\n").split(delimiter)
    if existing_first_fields[: len(header)] == header:
        print(f"[skip] {filename} already has header {header!r}")
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        rows = list(reader)

    if strip_trailing_empty_field:
        rows = [row[:-1] if row and row[-1] == "" else row for row in rows]

    out_path = RAW_DIR / filename
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[fixed] {filename}: wrote header {header!r}, {len(rows)} data rows")


def main() -> None:
    # Comma-delimited abusive-category files: label,text
    restore_header(
        "Abusive_ Tamil Text (Train dataset).csv",
        delimiter=",",
        header=["label", "text"],
    )
    restore_header(
        "Abusive_ Tamil Text (Test dataset).csv",
        delimiter=",",
        header=["label", "text"],
    )

    # Tab-delimited offensive files: label\ttext
    restore_header(
        "tamil_offensive_full.csv",
        delimiter="\t",
        header=["label", "text"],
    )

    # Tab-delimited offensive splits: text\tlabel (drop stray trailing tab)
    for name in (
        "tamil_offensive_full_train.csv",
        "tamil_offensive_full_dev.csv",
        "tamil_offensive_full_test.csv",
    ):
        restore_header(
            name,
            delimiter="\t",
            header=["text", "label"],
            strip_trailing_empty_field=True,
        )


if __name__ == "__main__":
    main()
