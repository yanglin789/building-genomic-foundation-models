#!/usr/bin/env python3
"""Build and randomly initialize a BERT-base masked language model for DNA.

The saved directory is compatible with Hugging Face ``from_pretrained`` and
contains both the model configuration and initialized model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM, set_seed


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer from the command line."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def probability(value: str) -> float:
    """Parse a probability in the closed interval [0, 1]."""
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a BERT-base DNA masked language model, randomly initialize "
            "its weights, and save a complete Hugging Face model directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        "--save-dir",
        dest="save_dir",
        type=Path,
        required=True,
        help="Directory in which to save config.json and model weights.",
    )
    parser.add_argument(
        "--vocab_size",
        "--vocab-size",
        dest="vocab_size",
        type=positive_integer,
        default=9,
        help="Tokenizer vocabulary size; 9 matches [PAD]/[UNK]/[CLS]/[SEP]/[MASK]/A/T/G/C.",
    )
    parser.add_argument(
        "--max_position_embeddings",
        "--max-position-embeddings",
        dest="max_position_embeddings",
        type=positive_integer,
        default=521,
        help="Maximum number of input token positions supported by the model.",
    )
    parser.add_argument(
        "--type_vocab_size",
        "--type-vocab-size",
        dest="type_vocab_size",
        type=positive_integer,
        default=1,
        help="Number of token-type embeddings; one is sufficient for single DNA sequences.",
    )
    parser.add_argument(
        "--pad_token_id",
        "--pad-token-id",
        dest="pad_token_id",
        type=int,
        default=0,
        help="Padding-token ID in the DNA tokenizer.",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        "--hidden-dropout-prob",
        dest="hidden_dropout_prob",
        type=probability,
        default=0.1,
        help="Dropout probability for hidden representations.",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        "--attention-probs-dropout-prob",
        dest="attention_probs_dropout_prob",
        type=probability,
        default=0.1,
        help="Dropout probability for attention probabilities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used to initialize model weights reproducibly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow saving into an existing non-empty directory.",
    )
    parser.add_argument(
        "--safe_serialization",
        "--safe-serialization",
        dest="safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save weights as model.safetensors instead of pytorch_model.bin.",
    )
    return parser


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def build_model(args: argparse.Namespace) -> BertForMaskedLM:
    """Create a standard BERT-base masked language model with random weights."""
    if not 0 <= args.pad_token_id < args.vocab_size:
        raise ValueError(
            f"--pad-token-id must be within [0, {args.vocab_size - 1}], "
            f"got {args.pad_token_id}"
        )

    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=args.pad_token_id,
        position_embedding_type="absolute",
        use_cache=True,
        model_type="bert",
    )

    # BertForMaskedLM initializes all model parameters when instantiated.
    return BertForMaskedLM(config)


def main() -> int:
    args = build_parser().parse_args()
    save_dir = args.save_dir.resolve()

    if save_dir.exists() and not save_dir.is_dir():
        raise ValueError(f"--save-dir exists but is not a directory: {save_dir}")
    if save_dir.exists() and any(save_dir.iterdir()) and not args.overwrite:
        raise ValueError(
            f"save directory is not empty: {save_dir}; pass --overwrite to replace model files"
        )

    set_seed(args.seed)
    model = build_model(args)
    total_parameters, trainable_parameters = count_parameters(model)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        save_dir,
        safe_serialization=args.safe_serialization,
    )

    summary = {
        "save_dir": str(save_dir),
        "model_class": model.__class__.__name__,
        "task": "masked_language_modeling",
        "architecture": "bert-base",
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "intermediate_size": model.config.intermediate_size,
        "type_vocab_size": model.config.type_vocab_size,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "seed": args.seed,
        "weight_file": (
            "model.safetensors" if args.safe_serialization else "pytorch_model.bin"
        ),
    }

    with (save_dir / "model_build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

