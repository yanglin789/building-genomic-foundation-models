#!/usr/bin/env python3
"""Generate a deterministic synthetic reference genome in FASTA format.

The generator is intentionally dependency-free so it can be used to bootstrap
the rest of the example pipeline before any machine-learning packages are
installed.  GC content is defined among non-``N`` bases, while ``n_ratio`` is
defined over all generated bases.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


DNA_BASES = ("A", "T", "G", "C", "N")


def probability(value: str) -> float:
    """Argparse type for a floating-point probability in the closed [0, 1] range."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 (inclusive)")
    return number


def positive_integer(value: str) -> int:
    """Argparse type for strictly positive integers."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reproducible synthetic FASTA reference. Files ending in "
            ".gz are compressed automatically."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination FASTA path (for example data/raw/demo_reference.fa).",
    )
    parser.add_argument(
        "--species-name",
        default="Synthetic species",
        help="Species name written into every FASTA header.",
    )
    parser.add_argument(
        "--num-sequences",
        type=positive_integer,
        default=5,
        help="Number of chromosomes/contigs to generate.",
    )
    parser.add_argument(
        "--sequence-length",
        type=positive_integer,
        default=100_000,
        help="Length in bases of each generated sequence.",
    )
    parser.add_argument(
        "--gc-content",
        type=probability,
        default=0.42,
        help="Target G+C fraction among bases that are not N.",
    )
    parser.add_argument(
        "--n-ratio",
        type=probability,
        default=0.01,
        help="Target fraction of unknown bases represented by N.",
    )
    parser.add_argument(
        "--line-width",
        type=positive_integer,
        default=80,
        help="Number of bases per FASTA sequence line.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed; identical arguments produce byte-identical content.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[TextIO]:
    """Open text while making gzip output reproducible across repeated runs."""
    if path.suffix.lower() == ".gz" and mode == "wt":
        # ``gzip.open`` records the current time and may embed the destination
        # filename. Suppress both fields so identical inputs are byte-identical.
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


def _safe_identifier(species_name: str) -> str:
    """Create a portable FASTA identifier prefix from a display name."""
    identifier = re.sub(r"[^A-Za-z0-9]+", "_", species_name.strip()).strip("_")
    return identifier or "synthetic_species"


def generate_fasta(
    output_path: Path,
    species_name: str,
    num_sequences: int,
    sequence_length: int,
    gc_content: float,
    n_ratio: float,
    line_width: int,
    seed: int,
) -> dict[str, int | float | str]:
    """Write the FASTA file and return realized base-composition statistics."""
    rng = random.Random(seed)
    non_n = 1.0 - n_ratio
    weights = (
        non_n * (1.0 - gc_content) / 2.0,  # A
        non_n * (1.0 - gc_content) / 2.0,  # T
        non_n * gc_content / 2.0,  # G
        non_n * gc_content / 2.0,  # C
        n_ratio,  # N
    )
    counts = {base: 0 for base in DNA_BASES}
    identifier = _safe_identifier(species_name)
    # Keep FASTA headers on one line even if a user-provided name contains tabs.
    header_species = " ".join(species_name.split()).replace('"', "'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(output_path, "wt") as handle:
        for sequence_index in range(1, num_sequences + 1):
            record_id = f"{identifier}_seq{sequence_index:06d}"
            handle.write(
                f'>{record_id} species="{header_species}" '
                f"length={sequence_length} seed={seed}\n"
            )

            # Generate one FASTA line at a time to keep memory bounded even for
            # chromosome-scale demonstrations.
            remaining = sequence_length
            while remaining:
                current_width = min(line_width, remaining)
                line = "".join(rng.choices(DNA_BASES, weights=weights, k=current_width))
                handle.write(line + "\n")
                for base in DNA_BASES:
                    counts[base] += line.count(base)
                remaining -= current_width

    total_bases = sum(counts.values())
    called_bases = total_bases - counts["N"]
    realized_gc = (
        (counts["G"] + counts["C"]) / called_bases if called_bases else 0.0
    )
    return {
        "output_path": str(output_path),
        "num_sequences": num_sequences,
        "total_bases": total_bases,
        "realized_gc_content": round(realized_gc, 6),
        "realized_n_ratio": round(counts["N"] / total_bases, 6),
        "seed": seed,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.species_name.strip():
        parser.error("--species-name must not be empty")
    if args.output_path.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {args.output_path}; pass --overwrite to replace it"
        )

    summary = generate_fasta(
        output_path=args.output_path,
        species_name=args.species_name,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        gc_content=args.gc_content,
        n_ratio=args.n_ratio,
        line_width=args.line_width,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

