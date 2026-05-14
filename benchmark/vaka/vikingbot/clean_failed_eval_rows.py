from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_INPUT = str(SCRIPT_DIR / "result" / "vaka_qa_result.csv")


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def is_blank_row(row: dict[str | None, str | list[str] | None]) -> bool:
    values: list[str] = []
    for value in row.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return not any(value.strip() for value in values)


def is_zero_token_row(row: dict[str | None, str | list[str] | None]) -> bool:
    value = row.get("response_input_tokens")
    if isinstance(value, list):
        return False
    text = (value or "").strip()
    if not text:
        return False
    try:
        return float(text) == 0
    except ValueError:
        return False


def clean_csv(path: Path, dry_run: bool) -> None:
    if not path.exists():
        print(f"Error: input file not found: {path}")
        raise SystemExit(1)

    raise_csv_field_limit()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print(f"Error: CSV has no header: {path}")
            raise SystemExit(1)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept_rows = []
    removed_blank = 0
    removed_zero_tokens = 0
    removed_indices: list[str] = []

    for row in rows:
        if is_blank_row(row):
            removed_blank += 1
            continue
        if is_zero_token_row(row):
            removed_zero_tokens += 1
            index = (row.get("question_index") or "").strip()
            if index:
                removed_indices.append(index)
            continue
        kept_rows.append(row)

    total_removed = removed_blank + removed_zero_tokens
    print(f"Input: {path}")
    print(f"Rows read: {len(rows)}")
    print(f"Rows kept: {len(kept_rows)}")
    print(f"Removed blank rows: {removed_blank}")
    print(f"Removed rows with response_input_tokens == 0: {removed_zero_tokens}")
    if removed_indices:
        print(f"Removed question_index values: {', '.join(removed_indices)}")

    if dry_run:
        print("Dry run only; CSV was not modified.")
        return

    if total_removed == 0:
        print("No rows matched the cleanup criteria; CSV was not modified.")
        return

    temp_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            writer = csv.DictWriter(f=temp_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Cleaned CSV written in place: {path}")
    print("question_index values were preserved; only failed/blank physical rows were removed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove failed Vaka eval rows from a result CSV in place."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to result CSV, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cleanup statistics without modifying the CSV.",
    )
    args = parser.parse_args()

    clean_csv(Path(args.input).expanduser(), args.dry_run)


if __name__ == "__main__":
    main()
