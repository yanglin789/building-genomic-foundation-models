#!/usr/bin/env python3
"""Convert FASTA records into a clean, fixed-window DNA pretraining corpus.

The output is plain text with one DNA window per line.  Windows never cross a
FASTA record boundary.  With ``--ambiguous-strategy split``, they also never
cross an ambiguous base, which is useful when ``N`` denotes an assembly gap.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


CANONICAL_BASES = frozenset("ATGC")


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean a FASTA file and write one fixed-length ATGC window per line. "
            "Input/output paths ending in .gz are handled transparently."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input FASTA path.")
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output plain-text corpus path; each non-empty line is one sample.",
    )
    parser.add_argument(
        "--window-size",
        type=positive_integer,
        default=512,
        help="Number of bases in every full output window.",
    )
    parser.add_argument(
        "--stride",
        type=positive_integer,
        default=None,
        help="Window step size; defaults to --window-size (non-overlapping windows).",
    )
    parser.add_argument(
        "--ambiguous-strategy",
        choices=("drop", "replace", "split", "error"),
        default="split",
        help=(
            "How to handle non-ATGC symbols: drop them, replace them, split the "
            "record at them, or fail immediately."
        ),
    )
    parser.add_argument(
        "--replacement-base",
        choices=tuple("ATGC"),
        default="A",
        help="Canonical base used when --ambiguous-strategy=replace.",
    )
    parser.add_argument(
        "--convert-u-to-t",
        dest="convert_u_to_t",
        action="store_true",
        default=True,
        help="Treat uracil U as thymine T before handling ambiguity.",
    )
    parser.add_argument(
        "--no-convert-u-to-t",
        dest="convert_u_to_t",
        action="store_false",
        help="Treat U as an ambiguous symbol.",
    )
    parser.add_argument(
        "--keep-remainder",
        action="store_true",
        help="Keep one final short window per cleaned segment instead of dropping it.",
    )
    parser.add_argument(
        "--min-remainder-length",
        type=positive_integer,
        default=1,
        help="Minimum length of a short window retained by --keep-remainder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[TextIO]:
    """Open plain/gzip text and make newly written gzip files deterministic."""
    if path.suffix.lower() == ".gz" and mode == "wt":
        with path.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="", fileobj=raw_handle, mode="wb", mtime=0
            ) as gzip_handle:
                with io.TextIOWrapper(
                    gzip_handle, encoding="utf-8", newline="\n"
                ) as text_handle:
                    yield text_handle
        return
    if path.suffix.lower() == ".gz":
        with gzip.open(path, mode, encoding="utf-8", newline="\n") as handle:
            yield handle
        return
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        yield handle


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(record_name, sequence)`` pairs from a FASTA file."""
    record_name: str | None = None
    sequence_parts: list[str] = []

    with _open_text(path, "rt") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_name is not None:
                    yield record_name, "".join(sequence_parts)
                record_name = line[1:].strip() or f"unnamed_record_at_line_{line_number}"
                sequence_parts = []
            else:
                if record_name is None:
                    raise ValueError(
                        f"sequence data found before the first FASTA header at line {line_number}"
                    )
                # Internal whitespace is ignored as a convenience for loosely
                # formatted FASTA exports.
                sequence_parts.append("".join(line.split()))

    if record_name is not None:
        yield record_name, "".join(sequence_parts)
    elif not sequence_parts:
        raise ValueError(f"no FASTA records found in {path}")


@dataclass(frozen=True)
class CleanedRecord:
    segments: tuple[str, ...]
    input_bases: int
    ambiguous_bases: int


def clean_sequence(
    sequence: str,
    record_name: str,
    strategy: str,
    replacement_base: str,
    convert_u_to_t: bool,
) -> CleanedRecord:
    """Normalize one sequence and return one or more canonical ATGC segments."""
    normalized = sequence.upper()
    if convert_u_to_t:
        normalized = normalized.replace("U", "T")
    ambiguous_count = sum(base not in CANONICAL_BASES for base in normalized)

    if strategy == "error" and ambiguous_count:
        position, symbol = next(
            (index, base)
            for index, base in enumerate(normalized, start=1)
            if base not in CANONICAL_BASES
        )
        raise ValueError(
            f"record {record_name!r} contains ambiguous symbol {symbol!r} "
            f"at 1-based position {position}"
        )

    if strategy == "drop":
        segments = ("".join(base for base in normalized if base in CANONICAL_BASES),)
    elif strategy == "replace":
        segments = (
            "".join(
                base if base in CANONICAL_BASES else replacement_base
                for base in normalized
            ),
        )
    elif strategy == "split":
        segments = tuple(match.group(0) for match in re.finditer(r"[ATGC]+", normalized))
    else:  # error with no ambiguous symbols
        segments = (normalized,)

    return CleanedRecord(
        segments=tuple(segment for segment in segments if segment),
        input_bases=len(normalized),
        ambiguous_bases=ambiguous_count,
    )


def iter_windows(
    sequence: str,
    window_size: int,
    stride: int,
    keep_remainder: bool,
    min_remainder_length: int,
) -> Iterator[str]:
    """Yield all full windows and, optionally, the next incomplete window."""
    if len(sequence) < window_size:
        if keep_remainder and len(sequence) >= min_remainder_length:
            yield sequence
        return

    last_start = -stride
    for start in range(0, len(sequence) - window_size + 1, stride):
        yield sequence[start : start + window_size]
        last_start = start

    remainder_start = last_start + stride
    remainder = sequence[remainder_start:]
    if (
        keep_remainder
        and remainder_start < len(sequence)
        and min_remainder_length <= len(remainder) < window_size
    ):
        yield remainder


def convert_fasta(
    input_path: Path,
    output_path: Path,
    window_size: int,
    stride: int,
    ambiguous_strategy: str,
    replacement_base: str,
    convert_u_to_t: bool,
    keep_remainder: bool,
    min_remainder_length: int,
) -> dict[str, int | str]:
    """Convert all records and return corpus-level processing statistics."""
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size")

    stats = {
        "records": 0,
        "input_bases": 0,
        "ambiguous_bases": 0,
        "cleaned_bases": 0,
        "full_windows": 0,
        "remainder_windows": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_suffix = ".tmp.gz" if output_path.suffix.lower() == ".gz" else ".tmp"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=temporary_suffix,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with _open_text(temporary_path, "wt") as output_handle:
            for record_name, raw_sequence in iter_fasta(input_path):
                stats["records"] += 1
                cleaned = clean_sequence(
                    raw_sequence,
                    record_name=record_name,
                    strategy=ambiguous_strategy,
                    replacement_base=replacement_base,
                    convert_u_to_t=convert_u_to_t,
                )
                stats["input_bases"] += cleaned.input_bases
                stats["ambiguous_bases"] += cleaned.ambiguous_bases

                for segment in cleaned.segments:
                    stats["cleaned_bases"] += len(segment)
                    for window in iter_windows(
                        segment,
                        window_size=window_size,
                        stride=stride,
                        keep_remainder=keep_remainder,
                        min_remainder_length=min_remainder_length,
                    ):
                        output_handle.write(window + "\n")
                        key = (
                            "full_windows"
                            if len(window) == window_size
                            else "remainder_windows"
                        )
                        stats[key] += 1

        if stats["records"] == 0:
            raise ValueError(f"no FASTA records found in {input_path}")
        output_windows = stats["full_windows"] + stats["remainder_windows"]
        if output_windows == 0:
            raise ValueError(
                "no output windows were generated; reduce --window-size or "
                "enable --keep-remainder"
            )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "window_size": window_size,
        "stride": stride,
        **stats,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stride = args.stride or args.window_size

    if not args.input_path.is_file():
        parser.error(f"input FASTA does not exist or is not a file: {args.input_path}")
    if args.input_path.resolve() == args.output_path.resolve():
        parser.error("--input-path and --output-path must be different")
    if args.output_path.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {args.output_path}; pass --overwrite to replace it"
        )
    if args.min_remainder_length > args.window_size:
        parser.error("--min-remainder-length cannot exceed --window-size")
    if stride > args.window_size:
        parser.error("--stride cannot exceed --window-size")

    try:
        summary = convert_fasta(
            input_path=args.input_path,
            output_path=args.output_path,
            window_size=args.window_size,
            stride=stride,
            ambiguous_strategy=args.ambiguous_strategy,
            replacement_base=args.replacement_base,
            convert_u_to_t=args.convert_u_to_t,
            keep_remainder=args.keep_remainder,
            min_remainder_length=args.min_remainder_length,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

