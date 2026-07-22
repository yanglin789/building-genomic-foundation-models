#!/usr/bin/env python3
"""Generate deterministic CSV datasets for supervised DNA fine-tuning.

Each output file contains exactly two columns::

    sequence,label

Binary classification uses one user-specified motif. Positive sequences contain
that motif exactly once, while negative sequences contain no occurrence of it.
Multilabel targets are stored as comma-separated multi-hot strings; Python's CSV
writer automatically surrounds those fields with double quotes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


DNA_ALPHABET = "ATGC"
TASK_TYPES = (
    "binary_classification",
    "multiclass_classification",
    "regression",
    "multilabel_classification",
)
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_SEED_OFFSETS = {"train": 11, "validation": 23, "test": 37}


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def probability(value: str) -> float:
    number = finite_float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def split_ratio(value: str) -> tuple[float, float, float]:
    try:
        ratios = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must contain three comma-separated numbers, for example 0.8,0.1,0.1"
        ) from error
    if len(ratios) != 3:
        raise argparse.ArgumentTypeError(
            "must contain exactly three values: train,validation,test"
        )
    if any(not math.isfinite(item) or item <= 0 for item in ratios):
        raise argparse.ArgumentTypeError("all split ratios must be positive finite numbers")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise argparse.ArgumentTypeError("split ratios must sum to 1.0")
    return ratios  # type: ignore[return-value]


def dna_string(value: str) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise argparse.ArgumentTypeError("DNA motif must not be empty")
    invalid = sorted(set(sequence) - set(DNA_ALPHABET))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"DNA motif may contain only {DNA_ALPHABET}; invalid symbols: {invalid}"
        )
    return sequence


def motif_list(value: str) -> list[str]:
    motifs = [dna_string(item) for item in value.split(",") if item.strip()]
    if not motifs:
        raise argparse.ArgumentTypeError("--motifs must contain at least one motif")
    if len(set(motifs)) != len(motifs):
        raise argparse.ArgumentTypeError("--motifs must not contain duplicate motifs")
    return motifs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic train/validation/test CSV files containing "
            "only sequence and label columns."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--num-sequences", type=positive_integer, default=10_000)
    parser.add_argument("--sequence-length", type=positive_integer, default=200)
    parser.add_argument(
        "--split-ratio",
        type=split_ratio,
        default=(0.8, 0.1, 0.1),
        help="Train, validation and test proportions.",
    )
    parser.add_argument(
        "--task-type",
        choices=TASK_TYPES,
        default="binary_classification",
    )

    parser.add_argument(
        "--positive-fraction",
        "--positive-rate",
        dest="positive_fraction",
        type=probability,
        default=0.50,
        help=(
            "Positive fraction for binary data, or independent per-label positive "
            "probability for multilabel data."
        ),
    )
    parser.add_argument(
        "--motif",
        type=dna_string,
        default="ATGCGTAC",
        help="Motif inserted exactly once into binary positive sequences.",
    )
    parser.add_argument(
        "--motifs",
        type=motif_list,
        default=None,
        help=(
            "Comma-separated motifs for multiclass or multilabel tasks. If omitted, "
            "distinct motifs are generated automatically."
        ),
    )
    parser.add_argument(
        "--num-labels",
        type=positive_integer,
        default=None,
        help="Required output count for multiclass or multilabel tasks unless --motifs is supplied.",
    )
    parser.add_argument(
        "--motif-length",
        type=positive_integer,
        default=8,
        help="Length used when multiclass or multilabel motifs are generated automatically.",
    )
    parser.add_argument("--background-gc", type=probability, default=0.50)

    parser.add_argument("--regression-min-gc", type=probability, default=0.20)
    parser.add_argument("--regression-max-gc", type=probability, default=0.80)
    parser.add_argument(
        "--regression-noise-std",
        type=finite_float,
        default=0.01,
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output CSV files to be replaced.",
    )
    return parser


def allocate_counts(total: int, proportions: Sequence[float]) -> list[int]:
    """Allocate an integer total with largest-remainder apportionment."""
    raw = [total * proportion for proportion in proportions]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def random_dna(rng: random.Random, length: int, gc_fraction: float) -> str:
    weights = (
        (1.0 - gc_fraction) / 2.0,  # A
        (1.0 - gc_fraction) / 2.0,  # T
        gc_fraction / 2.0,          # G
        gc_fraction / 2.0,          # C
    )
    return "".join(rng.choices(DNA_ALPHABET, weights=weights, k=length))


def make_binary_sequence(
    rng: random.Random,
    length: int,
    gc_fraction: float,
    motif: str,
    positive: bool,
) -> str:
    """Generate a sequence with exactly one motif for positives and none for negatives."""
    for _ in range(10_000):
        background = random_dna(rng, length, gc_fraction)
        if motif in background:
            continue
        if not positive:
            return background

        start = rng.randrange(length - len(motif) + 1)
        sequence = background[:start] + motif + background[start + len(motif) :]
        if sequence.count(motif) == 1:
            return sequence
    raise RuntimeError(
        "failed to generate an exact binary motif pattern; use a longer motif or sequence"
    )


def _longest_run(sequence: str) -> int:
    longest = current = 1
    for left, right in zip(sequence, sequence[1:]):
        current = current + 1 if left == right else 1
        longest = max(longest, current)
    return longest


def build_distinct_motifs(count: int, length: int, seed: int) -> list[str]:
    """Build reproducible motifs with moderate GC and pairwise separation."""
    rng = random.Random(seed ^ 0x5F3759DF)
    motifs: list[str] = []
    minimum_distance = max(2, length // 3)
    for _ in range(100_000):
        candidate = "".join(rng.choice(DNA_ALPHABET) for _ in range(length))
        gc = (candidate.count("G") + candidate.count("C")) / length
        if not 0.25 <= gc <= 0.75 or _longest_run(candidate) > 3:
            continue
        if any(
            sum(a != b for a, b in zip(candidate, existing)) < minimum_distance
            for existing in motifs
        ):
            continue
        motifs.append(candidate)
        if len(motifs) == count:
            return motifs
    raise ValueError(
        "could not construct enough distinct motifs; increase --motif-length or reduce --num-labels"
    )


def make_exact_motif_sequence(
    rng: random.Random,
    length: int,
    gc_fraction: float,
    motifs: Sequence[str],
    active_labels: Iterable[int],
) -> str:
    """Generate a sequence containing exactly the motifs selected by active_labels."""
    active = sorted(set(active_labels))
    for _ in range(10_000):
        background = random_dna(rng, length, gc_fraction)
        if any(motif in background for motif in motifs):
            continue

        sequence = list(background)
        occupied: list[tuple[int, int]] = []
        failed = False
        for label_index in active:
            motif = motifs[label_index]
            starts = [
                start
                for start in range(length - len(motif) + 1)
                if all(
                    start + len(motif) <= used_start or start >= used_end
                    for used_start, used_end in occupied
                )
            ]
            if not starts:
                failed = True
                break
            start = rng.choice(starts)
            sequence[start : start + len(motif)] = motif
            occupied.append((start, start + len(motif)))
        if failed:
            continue

        result = "".join(sequence)
        if all((motif in result) == (index in active) for index, motif in enumerate(motifs)):
            return result
    raise RuntimeError(
        "failed to generate the requested motif combination; increase sequence length"
    )


def binary_rows(
    split: str,
    size: int,
    positive_count: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[str, int]], dict[str, object]]:
    rng = random.Random(args.seed + SPLIT_SEED_OFFSETS[split])
    labels = [1] * positive_count + [0] * (size - positive_count)
    rng.shuffle(labels)
    rows = [
        (
            make_binary_sequence(
                rng,
                args.sequence_length,
                args.background_gc,
                args.motif,
                positive=bool(label),
            ),
            label,
        )
        for label in labels
    ]
    counts = Counter(labels)
    return rows, {"label_counts": {"0": counts[0], "1": counts[1]}}


def multiclass_rows(
    split: str,
    size: int,
    args: argparse.Namespace,
    motifs: Sequence[str],
) -> tuple[list[tuple[str, int]], dict[str, object]]:
    rng = random.Random(args.seed + SPLIT_SEED_OFFSETS[split])
    labels = [index % len(motifs) for index in range(size)]
    rng.shuffle(labels)
    rows = [
        (
            make_exact_motif_sequence(
                rng,
                args.sequence_length,
                args.background_gc,
                motifs,
                [label],
            ),
            label,
        )
        for label in labels
    ]
    counts = Counter(labels)
    return rows, {
        "label_counts": {str(index): counts[index] for index in range(len(motifs))}
    }


def multilabel_rows(
    split: str,
    size: int,
    args: argparse.Namespace,
    motifs: Sequence[str],
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    rng = random.Random(args.seed + SPLIT_SEED_OFFSETS[split])
    vectors = [
        [int(rng.random() < args.positive_fraction) for _ in motifs]
        for _ in range(size)
    ]
    # Ensure every label has both positive and negative examples in each nontrivial split.
    if size >= 2:
        for label_index in range(len(motifs)):
            column = [vector[label_index] for vector in vectors]
            if not any(column):
                vectors[rng.randrange(size)][label_index] = 1
            elif all(column):
                vectors[rng.randrange(size)][label_index] = 0

    rows: list[tuple[str, str]] = []
    for vector in vectors:
        active = [index for index, value in enumerate(vector) if value]
        sequence = make_exact_motif_sequence(
            rng,
            args.sequence_length,
            args.background_gc,
            motifs,
            active,
        )
        # csv.writer quotes this field automatically because it contains commas.
        rows.append((sequence, ",".join(str(value) for value in vector)))

    positives = [sum(vector[index] for vector in vectors) for index in range(len(motifs))]
    return rows, {
        "positive_counts": positives,
        "negative_counts": [size - value for value in positives],
    }


def regression_rows(
    split: str,
    size: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[str, float]], dict[str, object]]:
    rng = random.Random(args.seed + SPLIT_SEED_OFFSETS[split])
    rows: list[tuple[str, float]] = []
    labels: list[float] = []
    for _ in range(size):
        latent_gc = rng.uniform(args.regression_min_gc, args.regression_max_gc)
        sequence = random_dna(rng, args.sequence_length, latent_gc)
        realized_gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
        label = realized_gc + rng.gauss(0.0, args.regression_noise_std)
        label = round(min(1.0, max(0.0, label)), 6)
        rows.append((sequence, label))
        labels.append(label)
    return rows, {
        "label_min": min(labels),
        "label_max": max(labels),
        "label_mean": round(sum(labels) / len(labels), 6),
    }


def write_csv(path: Path, rows: Iterable[tuple[object, object]]) -> None:
    """Write sequence,label atomically using standards-compliant CSV quoting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sequence", "label"])
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_motifs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str] | None:
    if args.task_type == "binary_classification":
        return [args.motif]
    if args.task_type == "regression":
        return None

    if args.motifs is not None:
        motifs = args.motifs
        if args.num_labels is not None and args.num_labels != len(motifs):
            parser.error("--num-labels must match the number of values in --motifs")
        args.num_labels = len(motifs)
    else:
        if args.num_labels is None:
            parser.error(f"{args.task_type} requires --num-labels or --motifs")
        if args.task_type == "multiclass_classification" and args.num_labels < 3:
            parser.error("multiclass_classification requires --num-labels >= 3")
        if args.task_type == "multilabel_classification" and args.num_labels < 2:
            parser.error("multilabel_classification requires --num-labels >= 2")
        try:
            motifs = build_distinct_motifs(args.num_labels, args.motif_length, args.seed)
        except ValueError as error:
            parser.error(str(error))

    if any(len(motif) > args.sequence_length for motif in motifs):
        parser.error("every motif must be no longer than --sequence-length")
    return motifs


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    outputs = (args.train_output, args.validation_output, args.test_output)
    if len({path.resolve() for path in outputs}) != 3:
        parser.error("train, validation and test output paths must be different")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output file(s) already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    split_sizes = allocate_counts(args.num_sequences, args.split_ratio)
    if any(size == 0 for size in split_sizes):
        parser.error("--num-sequences is too small for the requested three-way split")

    if len(args.motif) > args.sequence_length:
        parser.error("--motif cannot be longer than --sequence-length")
    if args.task_type == "binary_classification" and not 0.0 <= args.positive_fraction <= 1.0:
        parser.error("--positive-fraction must be between 0 and 1")
    if args.task_type == "multilabel_classification" and not 0.0 < args.positive_fraction < 1.0:
        parser.error("multilabel_classification requires 0 < --positive-fraction < 1")
    if args.task_type == "regression":
        if args.regression_noise_std < 0:
            parser.error("--regression-noise-std must be non-negative")
        if args.regression_min_gc >= args.regression_max_gc:
            parser.error("--regression-min-gc must be smaller than --regression-max-gc")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    motifs = resolve_motifs(args, parser)

    split_sizes = allocate_counts(args.num_sequences, args.split_ratio)
    output_paths = dict(
        zip(SPLIT_NAMES, (args.train_output, args.validation_output, args.test_output))
    )

    # Apportion the requested total positives across splits exactly.
    positive_counts: dict[str, int] = {}
    if args.task_type == "binary_classification":
        total_positive = round(args.num_sequences * args.positive_fraction)
        allocated = allocate_counts(total_positive, [size / args.num_sequences for size in split_sizes])
        positive_counts = dict(zip(SPLIT_NAMES, allocated))

    summaries: dict[str, dict[str, object]] = {}
    generated: dict[str, list[tuple[object, object]]] = {}
    try:
        for split, size in zip(SPLIT_NAMES, split_sizes):
            if args.task_type == "binary_classification":
                rows, details = binary_rows(split, size, positive_counts[split], args)
            elif args.task_type == "multiclass_classification":
                assert motifs is not None
                rows, details = multiclass_rows(split, size, args, motifs)
            elif args.task_type == "multilabel_classification":
                assert motifs is not None
                rows, details = multilabel_rows(split, size, args, motifs)
            else:
                rows, details = regression_rows(split, size, args)
            generated[split] = rows
            summaries[split] = {"size": size, **details}
    except RuntimeError as error:
        parser.error(str(error))

    try:
        for split in SPLIT_NAMES:
            write_csv(output_paths[split], generated[split])
    except OSError as error:
        parser.error(f"could not write CSV output: {error}")

    summary = {
        "task_type": args.task_type,
        "num_sequences": args.num_sequences,
        "sequence_length": args.sequence_length,
        "split_ratio": list(args.split_ratio),
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items()},
        "columns": ["sequence", "label"],
        "seed": args.seed,
        "motif": args.motif if args.task_type == "binary_classification" else None,
        "motifs": list(motifs) if motifs is not None and args.task_type != "binary_classification" else None,
        "splits": summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

