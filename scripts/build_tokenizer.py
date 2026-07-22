#!/usr/bin/env python3
"""Build a lossless DNA tokenizer with Hugging Face Tokenizers.

The tokenizer vocabulary is generated exhaustively from the ATGC alphabet, so
no training corpus or input dataset is required. The serialized tokenizer uses
a WordLevel model. DNA strings are normalized and split in Python before being
passed to ``PreTrainedTokenizerFast`` because overlapping k-mers (``stride < k``)
cannot be represented by a serializable built-in Hugging Face pre-tokenizer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


LOGGER = logging.getLogger("build_tokenizer")
DNA_ALPHABET = "ATGC"
DNA_METADATA_FILENAME = "dna_tokenizer_config.json"
DNA_METADATA_VERSION = 1
SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}


def normalize_dna(sequence: str, ambiguous_policy: str = "error") -> str:
    """Uppercase DNA, remove whitespace, and handle non-ATGC characters.

    Args:
        sequence: A raw DNA string. Embedded spaces/newlines are allowed.
        ambiguous_policy: ``error`` rejects ambiguity; ``clean`` drops every
            character outside ATGC. Whitespace is always removed first.

    Returns:
        A non-empty string containing only A/T/G/C.
    """

    if ambiguous_policy not in {"error", "clean"}:
        raise ValueError(
            f"ambiguous_policy must be 'error' or 'clean', got {ambiguous_policy!r}"
        )
    if not isinstance(sequence, str):
        raise TypeError(f"DNA sequence must be str, got {type(sequence).__name__}")

    # DNA corpora are commonly wrapped over multiple lines, so whitespace is
    # formatting rather than biological content.
    compact = "".join(sequence.split()).upper()
    invalid = [
        (index, base)
        for index, base in enumerate(compact)
        if base not in DNA_ALPHABET
    ]
    if invalid and ambiguous_policy == "error":
        preview = ", ".join(f"{base!r}@{index}" for index, base in invalid[:8])
        if len(invalid) > 8:
            preview += f", ... ({len(invalid)} invalid characters total)"
        raise ValueError(f"DNA contains characters outside {DNA_ALPHABET}: {preview}")
    if invalid:
        compact = "".join(base for base in compact if base in DNA_ALPHABET)
    if not compact:
        raise ValueError("DNA sequence is empty after normalization")
    return compact


def split_dna_tokens(
    normalized_sequence: str,
    strategy: str,
    k: int,
    stride: int,
) -> list[str]:
    """Split normalized DNA into single bases or lossless (possibly overlapping) k-mers.

    For k-mers, windows start at ``0, stride, 2*stride, ...`` and the final
    windows may be shorter than ``k``.  Keeping these suffix fragments makes
    every input length round-trip exactly, including lengths not divisible by
    either ``k`` or ``stride``.
    """

    _validate_strategy(strategy, k, stride)
    if not normalized_sequence or any(base not in DNA_ALPHABET for base in normalized_sequence):
        raise ValueError("split_dna_tokens expects a non-empty, normalized ATGC sequence")
    if strategy == "single":
        return list(normalized_sequence)
    return [
        normalized_sequence[start : start + k]
        for start in range(0, len(normalized_sequence), stride)
    ]


def merge_dna_tokens(tokens: Sequence[str], strategy: str, k: int, stride: int) -> str:
    """Reconstruct DNA and verify that overlapping k-mers agree exactly."""

    _validate_strategy(strategy, k, stride)
    if not tokens:
        raise ValueError("Cannot decode an empty DNA token sequence")
    for token in tokens:
        if not token or len(token) > k or any(base not in DNA_ALPHABET for base in token):
            raise ValueError(f"Invalid DNA token encountered during decode: {token!r}")

    if strategy == "single":
        if any(len(token) != 1 for token in tokens):
            raise ValueError("Single-nucleotide tokenizer produced a multi-base token")
        return "".join(tokens)

    assembled: list[str] = []
    for token_index, token in enumerate(tokens):
        start = token_index * stride
        if start > len(assembled):
            raise ValueError(
                f"K-mer tokens leave a gap before token {token_index}: "
                f"start={start}, assembled_length={len(assembled)}"
            )
        for offset, base in enumerate(token):
            position = start + offset
            if position < len(assembled):
                if assembled[position] != base:
                    raise ValueError(
                        "Overlapping k-mers disagree at DNA position "
                        f"{position}: {assembled[position]!r} != {base!r}"
                    )
            else:
                assembled.append(base)
    return "".join(assembled)


def encode_dna(
    tokenizer: PreTrainedTokenizerFast,
    sequence: str,
    config: Mapping[str, Any],
    *,
    ambiguous_policy: str | None = None,
    add_special_tokens: bool = True,
    **tokenizer_kwargs: Any,
) -> Any:
    """Encode one raw DNA string using saved tokenizer strategy metadata.

    The returned object is the normal Hugging Face ``BatchEncoding``.  Callers
    can pass standard keyword arguments such as ``return_tensors='pt'``.
    """

    strategy, k, stride = _strategy_from_config(config)
    policy = ambiguous_policy or str(config.get("ambiguous_policy", "error"))
    normalized = normalize_dna(sequence, policy)
    dna_tokens = split_dna_tokens(normalized, strategy, k, stride)
    return tokenizer(
        dna_tokens,
        is_split_into_words=True,
        add_special_tokens=add_special_tokens,
        **tokenizer_kwargs,
    )


def decode_dna(
    tokenizer: PreTrainedTokenizerFast,
    token_ids: Any,
    config: Mapping[str, Any],
    *,
    skip_special_tokens: bool = True,
) -> str:
    """Decode IDs emitted by :func:`encode_dna` and validate k-mer overlaps."""

    strategy, k, stride = _strategy_from_config(config)
    ids = _flatten_token_ids(token_ids)
    special_ids = set(tokenizer.all_special_ids) if skip_special_tokens else set()
    dna_tokens: list[str] = []
    for token_id in ids:
        if token_id in special_ids:
            continue
        if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
            raise ValueError("Cannot decode DNA losslessly because [UNK] is present")
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None or token in tokenizer.all_special_tokens:
            raise ValueError(f"Unexpected non-DNA token id during decode: {token_id}")
        dna_tokens.append(token)
    return merge_dna_tokens(dna_tokens, strategy, k, stride)


def load_dna_tokenizer_config(tokenizer_dir: str | Path) -> dict[str, Any]:
    """Load and minimally validate ``dna_tokenizer_config.json``."""

    path = Path(tokenizer_dir) / DNA_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"DNA tokenizer metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format_version") != DNA_METADATA_VERSION:
        raise ValueError(
            f"Unsupported DNA tokenizer metadata version: {config.get('format_version')!r}"
        )
    _strategy_from_config(config)
    return config


def required_vocab_size(strategy: str, k: int) -> int:
    """Return exact exhaustive vocabulary size, including special tokens."""

    if strategy == "single":
        return len(SPECIAL_TOKENS) + len(DNA_ALPHABET)
    return len(SPECIAL_TOKENS) + sum(len(DNA_ALPHABET) ** length for length in range(1, k + 1))


def build_complete_vocab(strategy: str, k: int, max_vocab_size: int) -> dict[str, int]:
    """Build every possible ATGC token so valid DNA never maps to ``[UNK]``."""

    needed = required_vocab_size(strategy, k)
    if needed > max_vocab_size:
        raise ValueError(
            f"The requested tokenizer needs {needed:,} tokens, exceeding "
            f"--max-vocab-size={max_vocab_size:,}. Reduce k or raise the explicit limit."
        )
    vocab: dict[str, int] = {}
    for token in SPECIAL_TOKENS.values():
        vocab[token] = len(vocab)
    max_token_length = 1 if strategy == "single" else k
    for length in range(1, max_token_length + 1):
        for bases in itertools.product(DNA_ALPHABET, repeat=length):
            token = "".join(bases)
            vocab[token] = len(vocab)
    return vocab


def create_fast_tokenizer(
    vocab: Mapping[str, int],
    model_max_length: int,
) -> PreTrainedTokenizerFast:
    """Create the serializable Hugging Face fast-tokenizer backend."""

    backend = Tokenizer(
        models.WordLevel(
            vocab=dict(vocab),
            unk_token=SPECIAL_TOKENS["unk_token"],
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # ``decode_dna`` handles overlap-aware decoding. Fuse remains useful for
    # inspecting single-base and non-overlapping token streams.
    backend.decoder = decoders.Fuse()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKENS['cls_token']} $A {SPECIAL_TOKENS['sep_token']}",
        pair=(
            f"{SPECIAL_TOKENS['cls_token']} $A {SPECIAL_TOKENS['sep_token']} "
            f"$B:1 {SPECIAL_TOKENS['sep_token']}:1"
        ),
        special_tokens=[
            (SPECIAL_TOKENS["cls_token"], vocab[SPECIAL_TOKENS["cls_token"]]),
            (SPECIAL_TOKENS["sep_token"], vocab[SPECIAL_TOKENS["sep_token"]]),
        ],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
        **SPECIAL_TOKENS,
    )


def _validate_strategy(strategy: str, k: int, stride: int) -> None:
    if strategy not in {"single", "kmer"}:
        raise ValueError(f"Unknown DNA tokenization strategy: {strategy!r}")
    if k < 1 or stride < 1:
        raise ValueError("k and stride must both be positive integers")
    if strategy == "single" and (k != 1 or stride != 1):
        raise ValueError("single strategy requires k=1 and stride=1")
    if strategy == "kmer" and stride > k:
        raise ValueError("k-mer stride cannot exceed k; stride > k would lose bases")


def _strategy_from_config(config: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        strategy = str(config["tokenization_strategy"])
        k = int(config["k"])
        stride = int(config["stride"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("DNA tokenizer metadata lacks valid strategy/k/stride fields") from exc
    _validate_strategy(strategy, k, stride)
    return strategy, k, stride


def _flatten_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a one-dimensional list/array/tensor")
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise ValueError("decode_dna accepts exactly one encoded DNA sequence")
        token_ids = list(token_ids[0])
    try:
        return [int(token_id) for token_id in token_ids]
    except (TypeError, ValueError) as exc:
        raise TypeError("token_ids contains a non-integer value") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an exhaustive, lossless Hugging Face DNA tokenizer without "
            "requiring a training corpus."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Tokenizer output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--strategy",
        choices=("single", "kmer"),
        default="kmer",
        help="DNA tokenization strategy (default: kmer).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="K-mer length (kmer default: 6).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="K-mer start step; defaults to k. Must be <= k for lossless coverage.",
    )
    parser.add_argument(
        "--ambiguous-policy",
        choices=("error", "clean"),
        default="error",
        help="Runtime policy for non-ATGC characters in sequences passed to encode_dna().",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=100_000,
        help="Safety limit for the exhaustive vocabulary (default: 100000).",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=512,
        help="Saved tokenizer model_max_length (default: 512 tokens).",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_vocab_size < len(SPECIAL_TOKENS) + len(DNA_ALPHABET):
        raise ValueError("--max-vocab-size is too small for DNA and special tokens")
    if args.model_max_length < 2:
        raise ValueError("--model-max-length must be at least 2")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"--output-dir exists but is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not bool(getattr(args, "overwrite", False)):
            raise ValueError(
                f"output directory is not empty: {output_dir}; "
                "pass --overwrite to replace tokenizer files"
            )

    if args.strategy == "single":
        k = 1 if args.k is None else args.k
        stride = 1 if args.stride is None else args.stride
    else:
        k = 6 if args.k is None else args.k
        stride = k if args.stride is None else args.stride
    _validate_strategy(args.strategy, k, stride)

    vocab = build_complete_vocab(args.strategy, k, args.max_vocab_size)
    tokenizer = create_fast_tokenizer(vocab, args.model_max_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Unknown init kwargs are kept in the standard tokenizer_config.json too,
    # while the dedicated JSON below is the stable cross-script contract.
    tokenizer.init_kwargs.update(
        {
            "dna_tokenizer_metadata_file": DNA_METADATA_FILENAME,
            "dna_tokenization_strategy": args.strategy,
            "dna_k": k,
            "dna_stride": stride,
        }
    )
    tokenizer.save_pretrained(output_dir)

    metadata: dict[str, Any] = {
        "format_version": DNA_METADATA_VERSION,
        "tokenization_strategy": args.strategy,
        "k": k,
        "stride": stride,
        "alphabet": DNA_ALPHABET,
        "ambiguous_policy": args.ambiguous_policy,
        "case_policy": "uppercase",
        "whitespace_policy": "remove",
        "vocabulary_mode": "complete",
        "corpus_required": False,
        "vocab_size": len(vocab),
        "max_vocab_size": args.max_vocab_size,
        "model_max_length": args.model_max_length,
        "special_tokens": SPECIAL_TOKENS,
        "requires_pretokenization": True,
        "encoding_helper": "scripts/build_tokenizer.py::encode_dna",
        "decoding_helper": "scripts/build_tokenizer.py::decode_dna",
    }
    metadata_path = output_dir / DNA_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    LOGGER.info(
        "Saved %s DNA tokenizer: vocab=%d, k=%d, stride=%d, output=%s",
        args.strategy,
        len(vocab),
        k,
        stride,
        output_dir,
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run(parse_args(argv))
    except (OSError, ValueError, TypeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


