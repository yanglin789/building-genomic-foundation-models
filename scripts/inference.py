#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DNA_METADATA_NAME = "dna_tokenizer_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned DNA model.")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model directory or Hugging Face Hub model name.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="Predict a single DNA sequence.")
    source.add_argument("--input-file", type=Path, help="CSV/TXT file for batch prediction.")

    parser.add_argument("--sequence-column", default="sequence", help="Column name for sequences in CSV.")
    parser.add_argument("--output-file", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for multi-label classification.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_dna_config(model_path: str) -> dict[str, Any]:
    path = Path(model_path) / DNA_METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing DNA tokenizer config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_sequence(sequence: Any, config: dict[str, Any]) -> list[str]:
    sequence = "".join(str(sequence).split()).upper()
    alphabet = set(str(config.get("alphabet", "ATGC")).upper())
    invalid = sorted(set(sequence) - alphabet)

    if invalid and config.get("ambiguous_policy", "error") == "clean":
        sequence = "".join(base for base in sequence if base in alphabet)
    elif invalid:
        raise ValueError(f"DNA sequence contains invalid characters: {invalid}")
    if not sequence:
        raise ValueError("DNA sequence is empty.")

    strategy = config.get("tokenization_strategy", "single")
    if strategy == "single":
        return list(sequence)
    if strategy == "kmer":
        k = int(config.get("k", 6))
        stride = int(config.get("stride", k))
        return [sequence[i : i + k] for i in range(0, len(sequence), stride)]
    raise ValueError(f"Unsupported DNA tokenization strategy: {strategy}")


def load_sequences(args: argparse.Namespace) -> list[str]:
    if args.sequence is not None:
        return [args.sequence]

    if args.input_file.suffix.lower() == ".csv":
        with args.input_file.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or args.sequence_column not in rows[0]:
            raise ValueError(f"CSV must contain column {args.sequence_column!r}.")
        return [row[args.sequence_column] for row in rows]

    with args.input_file.open("r", encoding="utf-8") as handle:
        sequences = [line.strip() for line in handle if line.strip()]
    if not sequences:
        raise ValueError("No valid sequences found in input file.")
    return sequences


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please use --device cpu.")
    return torch.device(name)


def label_names(model: torch.nn.Module) -> dict[int, str]:
    mapping = getattr(model.config, "id2label", {}) or {}
    return {i: str(mapping.get(i, mapping.get(str(i), i))) for i in range(model.config.num_labels)}


def predict(args: argparse.Namespace, sequences: list[str]) -> list[dict[str, Any]]:
    dna_config = load_dna_config(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    device = get_device(args.device)
    model.to(device).eval()

    max_length = args.max_seq_length or int(dna_config.get("model_max_length", 512))
    names = label_names(model)
    problem_type = model.config.problem_type or (
        "regression" if model.config.num_labels == 1 else "single_label_classification"
    )
    results: list[dict[str, Any]] = []

    for start in range(0, len(sequences), args.batch_size):
        batch_sequences = sequences[start : start + args.batch_size]
        tokens = [prepare_sequence(sequence, dna_config) for sequence in batch_sequences]
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        if tokenizer.unk_token_id is not None and torch.any(inputs["input_ids"] == tokenizer.unk_token_id):
            raise ValueError("Tokenization result contains [UNK], please check DNA sequence or tokenizer configuration.")

        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits.float().cpu()

        if problem_type == "regression":
            for sequence, value in zip(batch_sequences, logits.squeeze(-1).tolist()):
                results.append({"sequence": sequence, "prediction": float(value)})

        elif problem_type == "multi_label_classification":
            probabilities = torch.sigmoid(logits)
            for sequence, row in zip(batch_sequences, probabilities):
                probs = {names[i]: float(row[i]) for i in range(len(names))}
                predicted = [name for name, probability in probs.items() if probability >= args.threshold]
                results.append(
                    {
                        "sequence": sequence,
                        "prediction": json.dumps(predicted, ensure_ascii=False),
                        "probabilities": json.dumps(probs, ensure_ascii=False),
                    }
                )

        else:
            probabilities = torch.softmax(logits, dim=-1)
            for sequence, row in zip(batch_sequences, probabilities):
                class_id = int(torch.argmax(row))
                probs = {names[i]: float(row[i]) for i in range(len(names))}
                results.append(
                    {
                        "sequence": sequence,
                        "prediction": names[class_id],
                        "confidence": float(row[class_id]),
                        "probabilities": json.dumps(probs, ensure_ascii=False),
                    }
                )
    return results


def save_results(results: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than 0.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    results = predict(args, load_sequences(args))
    if args.sequence is not None:
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
    else:
        save_results(results, args.output_file)
        print(f"Prediction complete: {len(results)} sequences, results saved to {args.output_file}")


if __name__ == "__main__":
    main()
