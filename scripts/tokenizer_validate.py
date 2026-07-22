#!/usr/bin/env python3
"""Strict round-trip validation for a tokenizer built by build_tokenizer.py."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from transformers import PreTrainedTokenizerFast

from build_tokenizer import (
    DNA_ALPHABET,
    decode_dna,
    encode_dna,
    load_dna_tokenizer_config,
    normalize_dna,
    split_dna_tokens,
)


LOGGER = logging.getLogger("tokenizer_validate")


def _default_sequences(k: int, strategy: str) -> list[str]:
    """Include short, long, and deliberately non-k-multiple DNA inputs."""

    motif = "ATGCGTAC"
    lengths = {1, 4, 7, max(2, k), max(3, k + 1), max(5, 2 * k + 1)}
    if strategy == "kmer":
        # Explicit invariant: at least one test length is not divisible by k.
        non_multiple = k + 1
        lengths.add(non_multiple)
    return [((motif * ((length // len(motif)) + 1))[:length]) for length in sorted(lengths)]


def _read_sequence_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Sequence test file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _record(checks: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    checks.append({"name": name, "passed": passed, **details})


def validate_tokenizer(
    tokenizer_dir: Path,
    sequences: Sequence[str],
    policy_override: str | None,
) -> dict[str, Any]:
    config = load_dna_tokenizer_config(tokenizer_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    checks: list[dict[str, Any]] = []

    _record(checks, "fast_backend", bool(tokenizer.is_fast), observed=bool(tokenizer.is_fast))
    _record(
        checks,
        "vocabulary_size",
        len(tokenizer) == int(config["vocab_size"]),
        expected=int(config["vocab_size"]),
        observed=len(tokenizer),
    )
    missing_specials = [name for name in config["special_tokens"] if getattr(tokenizer, name) is None]
    _record(checks, "special_tokens", not missing_specials, missing=missing_specials)

    strategy = str(config["tokenization_strategy"])
    k = int(config["k"])
    stride = int(config["stride"])
    policy = policy_override or str(config["ambiguous_policy"])

    # Validate both ambiguity branches regardless of the saved default.
    clean_probe = " aTn-CgX\n"
    try:
        cleaned = normalize_dna(clean_probe, "clean")
        _record(
            checks,
            "ambiguous_clean_policy",
            cleaned == "ATCG",
            expected="ATCG",
            observed=cleaned,
        )
    except Exception as exc:  # report all failures before returning non-zero
        _record(checks, "ambiguous_clean_policy", False, error=str(exc))

    try:
        normalize_dna("ATNC", "error")
    except ValueError:
        _record(checks, "ambiguous_error_policy", True, expected="ValueError")
    else:
        _record(checks, "ambiguous_error_policy", False, expected="ValueError")

    # Also exercise ambiguity handling through the full tokenizer, not only the
    # normalizer. ``clean`` must remain lossless after invalid bases are dropped.
    try:
        clean_encoding = encode_dna(
            tokenizer,
            clean_probe,
            config,
            ambiguous_policy="clean",
            add_special_tokens=True,
        )
        clean_decoded = decode_dna(tokenizer, clean_encoding["input_ids"], config)
        _record(
            checks,
            "ambiguous_clean_round_trip",
            clean_decoded == "ATCG",
            expected="ATCG",
            observed=clean_decoded,
        )
    except Exception as exc:
        _record(checks, "ambiguous_clean_round_trip", False, error=str(exc))

    try:
        encode_dna(tokenizer, "ATNC", config, ambiguous_policy="error")
    except ValueError:
        _record(checks, "ambiguous_error_encode", True, expected="ValueError")
    else:
        _record(checks, "ambiguous_error_encode", False, expected="ValueError")

    round_trip_count = 0
    for index, raw_sequence in enumerate(sequences, start=1):
        check_name = f"round_trip_{index}"
        try:
            expected = normalize_dna(raw_sequence, policy)
            dna_tokens = split_dna_tokens(expected, strategy, k, stride)
            encoded = encode_dna(
                tokenizer,
                raw_sequence,
                config,
                ambiguous_policy=policy,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            ids = encoded["input_ids"]
            has_unknown = tokenizer.unk_token_id in ids
            decoded = decode_dna(tokenizer, ids, config)
            passed = decoded == expected and not has_unknown
            _record(
                checks,
                check_name,
                passed,
                input=raw_sequence,
                normalized=expected,
                token_count=len(dna_tokens),
                encoded_id_count=len(ids),
                decoded=decoded,
                contains_unknown=has_unknown,
                non_k_multiple=(strategy == "kmer" and len(expected) % k != 0),
            )
            if passed:
                round_trip_count += 1
        except Exception as exc:
            _record(checks, check_name, False, input=raw_sequence, error=str(exc))

    if strategy == "kmer":
        exercised = any(
            check.get("non_k_multiple") is True and check.get("passed") is True
            for check in checks
        )
        _record(checks, "non_k_multiple_coverage", exercised, k=k)

    failed = [check for check in checks if not check["passed"]]
    return {
        "status": "passed" if not failed else "failed",
        "tokenizer_dir": str(tokenizer_dir.resolve()),
        "configuration": {
            "tokenization_strategy": strategy,
            "k": k,
            "stride": stride,
            "alphabet": DNA_ALPHABET,
            "ambiguous_policy_tested": policy,
            "vocab_size": len(tokenizer),
        },
        "summary": {
            "checks": len(checks),
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "round_trips": round_trip_count,
        },
        "checks": checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly validate a saved DNA tokenizer.")
    parser.add_argument("--tokenizer-dir", type=Path, required=True, help="Saved tokenizer directory.")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Additional raw DNA sequence; may be repeated.",
    )
    parser.add_argument(
        "--sequences-file",
        type=Path,
        default=None,
        help="Optional text file containing one additional test sequence per line.",
    )
    parser.add_argument(
        "--ambiguous-policy",
        choices=("metadata", "error", "clean"),
        default="metadata",
        help="Policy for user sequences; metadata uses the tokenizer's saved policy.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional JSON report output path; the report is always printed too.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        config = load_dna_tokenizer_config(args.tokenizer_dir)
        sequences = _default_sequences(int(config["k"]), str(config["tokenization_strategy"]))
        sequences.extend(args.sequence)
        if args.sequences_file is not None:
            sequences.extend(_read_sequence_file(args.sequences_file))
        policy = None if args.ambiguous_policy == "metadata" else args.ambiguous_policy
        report = validate_tokenizer(args.tokenizer_dir, sequences, policy)
        report_text = json.dumps(report, ensure_ascii=False, indent=2)
        print(report_text)
        if args.report_file is not None:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            with args.report_file.open("w", encoding="utf-8") as handle:
                handle.write(report_text)
                handle.write("\n")
        if report["status"] != "passed":
            LOGGER.error(
                "Tokenizer validation failed: %d/%d checks failed",
                report["summary"]["failed"],
                report["summary"]["checks"],
            )
            return 1
        LOGGER.info(
            "Tokenizer validation passed: %d checks, %d round trips",
            report["summary"]["checks"],
            report["summary"]["round_trips"],
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

