#!/usr/bin/env python
"""Initialize or continue pretraining a DNA masked language model from ATGC text.

The model can be initialized from scratch, optionally using a local model
configuration, or loaded from a complete local Hugging Face model directory to
inherit existing weights. Distributed execution can be launched with
``torchrun`` or ``accelerate launch``; ``Trainer`` reads the distributed
configuration from the environment.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch.distributed as dist

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger("dna_pretrain")
DNA_METADATA_NAME = "dna_tokenizer_config.json"


@dataclass(frozen=True)
class DnaTokenizationSpec:
    """Effective DNA pre-tokenization settings for this training run."""

    strategy: str
    k: int
    stride: int
    alphabet: str
    ambiguous_policy: str
    case_policy: str
    whitespace_policy: str
    metadata_path: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize and pretrain a DNA masked language model."
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Plain ATGC training text with one sequence per line.",
    )
    data.add_argument(
        "--validation-file",
        type=Path,
        help="Optional validation text; otherwise split from training data when evaluation is enabled.",
    )
    data.add_argument("--validation-split-percentage", type=float, default=5.0)
    data.add_argument("--max-train-samples", type=int)
    data.add_argument("--max-validation-samples", type=int)
    data.add_argument("--preprocessing-num-workers", type=int, default=1)
    data.add_argument("--dataset-cache-dir", type=Path)
    data.add_argument("--overwrite-cache", action="store_true")

    tok = parser.add_argument_group("tokenizer")
    tok.add_argument("--tokenizer-path", type=Path, required=True)
    tok.add_argument(
        "--tokenizer-metadata",
        type=Path,
        help=f"Defaults to {DNA_METADATA_NAME} under the tokenizer directory.",
    )
    tok.add_argument(
        "--tokenization-strategy",
        choices=("auto", "single", "kmer"),
        default="auto",
        help="Use tokenizer metadata with auto, or explicitly override its strategy.",
    )
    tok.add_argument("--kmer-size", type=int, help="Override metadata k.")
    tok.add_argument("--kmer-stride", type=int, help="Override metadata stride.")
    tok.add_argument(
        "--ambiguous-policy",
        choices=("metadata", "error", "clean"),
        default="metadata",
        help="Use metadata policy, reject non-ATGC characters, or remove them.",
    )
    tok.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum model input length including special tokens.",
    )
    tok.add_argument("--trust-remote-code", action="store_true")

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-init",
        choices=("from_scratch", "from_pretrained"),
        default="from_scratch",
        help=(
            "Model initialization mode. from_scratch randomly initializes weights; "
            "from_pretrained inherits weights from --model-path."
        ),
    )
    model.add_argument(
        "--model-path",
        type=Path,
        help=(
            "Local Hugging Face model directory. In from_scratch mode only its "
            "config is used; in from_pretrained mode both config and weights are loaded."
        ),
    )
    model.add_argument(
        "--config-name-or-path",
        help=(
            "Optional configuration name/path used only in from_scratch mode when "
            "--model-path is not supplied; otherwise create a BERT config from CLI values."
        ),
    )
    model.add_argument("--model-revision", default="main")
    model.add_argument("--hidden-size", type=int, default=256)
    model.add_argument("--num-hidden-layers", type=int, default=6)
    model.add_argument("--num-attention-heads", type=int, default=8)
    model.add_argument("--intermediate-size", type=int, default=1024)
    model.add_argument("--max-position-embeddings", type=int)
    model.add_argument("--mlm-probability", type=float, default=0.15)
    model.add_argument("--gradient-checkpointing", action="store_true")

    train = parser.add_argument_group("training")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--overwrite-output-dir", action="store_true")
    train.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint directory, or auto/last to use the newest checkpoint in output-dir.",
    )
    train.add_argument("--no-train", dest="do_train", action="store_false")
    train.add_argument("--no-eval", dest="do_eval", action="store_false")
    train.set_defaults(do_train=True, do_eval=True)
    train.add_argument("--num-train-epochs", type=float, default=3.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--per-device-train-batch-size", type=int, default=16)
    train.add_argument("--per-device-eval-batch-size", type=int, default=16)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--lr-scheduler-type", default="linear")
    train.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="epoch")
    train.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="epoch")
    train.add_argument("--logging-strategy", choices=("no", "steps", "epoch"), default="steps")
    train.add_argument("--logging-steps", type=int, default=50)
    train.add_argument("--eval-steps", type=int)
    train.add_argument("--save-steps", type=int, default=500)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--data-seed", type=int, default=42)
    train.add_argument(
        "--full-determinism",
        action="store_true",
        help="Enable deterministic Transformers/PyTorch algorithms (may reduce throughput).",
    )
    train.add_argument("--dataloader-num-workers", type=int, default=0)
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--tf32", action="store_true")
    train.add_argument("--use-cpu", action="store_true")
    train.add_argument("--optim", default="adamw_torch")
    train.add_argument("--report-to", nargs="*", default=["none"])
    train.add_argument("--run-name")
    train.add_argument("--deepspeed", help="Optional DeepSpeed JSON configuration path.")
    train.add_argument("--fsdp", help="Optional FSDP strategy string, for example 'full_shard auto_wrap'.")
    train.add_argument("--ddp-find-unused-parameters", action="store_true")
    train.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"DNA tokenizer metadata must be a JSON object: {path}")
    return payload


def load_tokenization_spec(args: argparse.Namespace) -> DnaTokenizationSpec:
    """Read tokenizer metadata and apply explicit command-line overrides."""

    metadata_path = args.tokenizer_metadata or (args.tokenizer_path / DNA_METADATA_NAME)
    metadata: Dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = _read_json(metadata_path)
    elif args.tokenization_strategy == "auto":
        raise FileNotFoundError(
            f"Cannot infer DNA tokenization: {metadata_path} does not exist. "
            "Pass --tokenizer-metadata or an explicit --tokenization-strategy."
        )
    else:
        metadata_path = None

    strategy = (
        metadata.get("tokenization_strategy", "single")
        if args.tokenization_strategy == "auto"
        else args.tokenization_strategy
    )
    strategy = str(strategy).lower()
    if strategy not in {"single", "kmer"}:
        raise ValueError(f"Unsupported tokenization strategy: {strategy!r}")

    if args.kmer_size is not None:
        k = int(args.kmer_size)
    elif "k" in metadata:
        k = int(metadata["k"])
    else:
        # Match build_tokenizer.py when k-mer tokenization is selected without metadata.
        k = 6 if strategy == "kmer" else 1

    if args.kmer_stride is not None:
        stride = int(args.kmer_stride)
    elif "stride" in metadata:
        stride = int(metadata["stride"])
    else:
        stride = k if strategy == "kmer" else 1

    if strategy == "single":
        k, stride = 1, 1
    if k < 1 or stride < 1 or stride > k:
        raise ValueError(f"Expected 1 <= stride <= k, got k={k}, stride={stride}.")

    ambiguous_policy = str(metadata.get("ambiguous_policy", "error")).lower()
    if args.ambiguous_policy != "metadata":
        ambiguous_policy = args.ambiguous_policy
    if ambiguous_policy not in {"error", "clean"}:
        raise ValueError(f"Unsupported ambiguous_policy: {ambiguous_policy!r}")

    return DnaTokenizationSpec(
        strategy=strategy,
        k=k,
        stride=stride,
        alphabet=str(metadata.get("alphabet", "ATGC")).upper(),
        ambiguous_policy=ambiguous_policy,
        case_policy=str(metadata.get("case_policy", "upper")).lower(),
        whitespace_policy=str(metadata.get("whitespace_policy", "remove")).lower(),
        metadata_path=metadata_path,
    )


def _source_metadata(spec: DnaTokenizationSpec) -> Dict[str, Any]:
    """Return a fresh copy of source metadata, or an empty object if absent."""

    return _read_json(spec.metadata_path) if spec.metadata_path is not None else {}


def validate_tokenizer(tokenizer: Any, spec: DnaTokenizationSpec) -> None:
    """Validate vocabulary and required special tokens before preprocessing."""

    metadata = _source_metadata(spec)
    expected_vocab_size = metadata.get("vocab_size")
    if expected_vocab_size is not None:
        try:
            expected_vocab_size = int(expected_vocab_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid vocab_size in DNA tokenizer metadata: {expected_vocab_size!r}"
            ) from exc
        if expected_vocab_size != len(tokenizer):
            raise ValueError(
                "Tokenizer/metadata vocabulary mismatch: "
                f"metadata vocab_size={expected_vocab_size}, loaded tokenizer size={len(tokenizer)}. "
                "Use metadata generated for this tokenizer."
            )

    required_tokens = (
        "pad_token",
        "unk_token",
        "cls_token",
        "sep_token",
        "mask_token",
    )
    seen_ids: Dict[int, str] = {}
    for token_name in required_tokens:
        token = getattr(tokenizer, token_name, None)
        token_id = getattr(tokenizer, f"{token_name}_id", None)
        if token is None or token_id is None:
            raise ValueError(
                f"The tokenizer must define {token_name} and {token_name}_id for DNA MLM pretraining."
            )
        if not 0 <= int(token_id) < len(tokenizer):
            raise ValueError(
                f"Tokenizer {token_name}_id={token_id} is outside vocabulary size {len(tokenizer)}."
            )
        mapped_id = tokenizer.convert_tokens_to_ids(token)
        if mapped_id != token_id:
            raise ValueError(
                f"Tokenizer {token_name}={token!r} maps to id {mapped_id}, not declared id {token_id}."
            )
        if int(token_id) in seen_ids:
            raise ValueError(
                f"Tokenizer special tokens {seen_ids[int(token_id)]} and {token_name} "
                f"share id {token_id}; distinct MLM special tokens are required."
            )
        seen_ids[int(token_id)] = token_name

    metadata_special_tokens = metadata.get("special_tokens")
    if metadata_special_tokens is not None:
        if not isinstance(metadata_special_tokens, Mapping):
            raise ValueError("DNA tokenizer metadata special_tokens must be a JSON object.")
        for token_name in required_tokens:
            expected_token = metadata_special_tokens.get(token_name)
            if expected_token is not None and expected_token != getattr(tokenizer, token_name):
                raise ValueError(
                    f"Tokenizer/metadata special-token mismatch for {token_name}: "
                    f"metadata={expected_token!r}, tokenizer={getattr(tokenizer, token_name)!r}."
                )


def build_effective_metadata(
    spec: DnaTokenizationSpec,
    tokenizer: Any,
    model_max_length: int,
) -> Dict[str, Any]:
    """Preserve source fields while recording the settings actually used."""

    metadata = _source_metadata(spec)
    metadata["format_version"] = int(metadata.get("format_version", 1))
    metadata["special_tokens"] = {
        token_name: getattr(tokenizer, token_name)
        for token_name in ("pad_token", "unk_token", "cls_token", "sep_token", "mask_token")
    }
    metadata.update(
        {
            "tokenization_strategy": spec.strategy,
            "k": spec.k,
            "stride": spec.stride,
            "alphabet": spec.alphabet,
            "ambiguous_policy": spec.ambiguous_policy,
            "case_policy": spec.case_policy,
            "whitespace_policy": spec.whitespace_policy,
            "vocab_size": len(tokenizer),
            "model_max_length": model_max_length,
            "requires_pretokenization": True,
        }
    )
    return metadata


def save_tokenizer_artifacts(
    output_dir: Path,
    tokenizer: Any,
    spec: DnaTokenizationSpec,
    model_max_length: int,
) -> None:
    """Save the tokenizer and an effective DNA metadata contract."""

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    metadata = build_effective_metadata(spec, tokenizer, model_max_length)
    target = output_dir / DNA_METADATA_NAME
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def normalize_dna(value: Any, spec: DnaTokenizationSpec) -> str:
    """Apply case, whitespace, and ambiguous-base policies from metadata."""

    if value is None:
        return ""
    sequence = str(value)
    if spec.whitespace_policy in {"remove", "strip", "clean"}:
        sequence = "".join(sequence.split())
    elif any(char.isspace() for char in sequence):
        raise ValueError("DNA sequence contains whitespace but whitespace_policy is not 'remove'.")

    if spec.case_policy in {"upper", "uppercase", "normalize", "insensitive"}:
        sequence = sequence.upper()
    alphabet = set(spec.alphabet)
    invalid = sorted(set(sequence) - alphabet)
    if invalid and spec.ambiguous_policy == "error":
        raise ValueError(f"DNA sequence contains characters outside {spec.alphabet}: {invalid}")
    if invalid:
        sequence = "".join(char for char in sequence if char in alphabet)
    return sequence


def dna_to_tokens(sequence: str, spec: DnaTokenizationSpec) -> List[str]:
    """Split DNA into single-base or sliding k-mer tokens, retaining the tail."""

    if spec.strategy == "single":
        return list(sequence)
    return [sequence[start : start + spec.k] for start in range(0, len(sequence), spec.stride)]


def _raise_for_unknown_tokens(
    encoded: Mapping[str, Any],
    token_lists: Sequence[Sequence[str]],
    tokenizer: Any,
) -> None:
    """Reject tokenizer/strategy mismatches instead of silently training on UNK."""

    unk_token_id = tokenizer.unk_token_id
    if unk_token_id is None:
        raise ValueError("The tokenizer must define unk_token_id so encoded DNA can be validated.")

    offenders: List[str] = []
    for batch_index, input_ids in enumerate(encoded["input_ids"]):
        positions = [index for index, token_id in enumerate(input_ids) if token_id == unk_token_id]
        if positions:
            token_preview = " ".join(token_lists[batch_index][:8])
            offenders.append(
                f"batch_index={batch_index}, positions={positions[:8]}, tokens={token_preview!r}"
            )
            if len(offenders) >= 3:
                break
    if offenders:
        raise ValueError(
            f"Tokenizer produced {tokenizer.unk_token or '[UNK]'} (id={unk_token_id}) for valid DNA "
            "after pre-tokenization. The tokenizer vocabulary and effective DNA metadata/CLI "
            "strategy are incompatible. Examples: "
            + "; ".join(offenders)
        )


def load_corpus(args: argparse.Namespace, spec: DnaTokenizationSpec, tokenizer: Any) -> DatasetDict:
    for path in (args.train_file, args.validation_file):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")

    data_files = {"train": str(args.train_file)}
    if args.validation_file is not None:
        data_files["validation"] = str(args.validation_file)
    raw = load_dataset(
        "text",
        data_files=data_files,
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
        keep_linebreaks=False,
    )

    if "validation" not in raw and args.do_eval:
        percentage = args.validation_split_percentage
        if not 0.0 < percentage < 100.0:
            raise ValueError("--validation-split-percentage must be between 0 and 100.")
        split = raw["train"].train_test_split(
            test_size=percentage / 100.0,
            seed=args.data_seed,
            shuffle=True,
        )
        raw = DatasetDict({"train": split["train"], "validation": split["test"]})
    elif "validation" not in raw:
        # With --no-eval, keep every example in the training split.
        raw = DatasetDict({"train": raw["train"]})

    def is_nonempty(example: Mapping[str, Any]) -> bool:
        return bool(normalize_dna(example["text"], spec))

    workers = args.preprocessing_num_workers if args.preprocessing_num_workers > 1 else None
    raw = DatasetDict(
        {
            split_name: dataset.filter(
                is_nonempty,
                num_proc=workers,
                desc=f"Filter empty {split_name} sequences",
            )
            for split_name, dataset in raw.items()
        }
    )

    def tokenize_batch(examples: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
        token_lists = [
            dna_to_tokens(normalize_dna(sequence, spec), spec) for sequence in examples["text"]
        ]
        # WordLevel DNA tokenizers must receive pre-split tokens so k-mer boundaries survive.
        encoded = tokenizer(
            token_lists,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
        )
        _raise_for_unknown_tokens(encoded, token_lists, tokenizer)
        return dict(encoded)

    tokenized = DatasetDict()
    for split_name, dataset in raw.items():
        tokenized[split_name] = dataset.map(
            tokenize_batch,
            batched=True,
            num_proc=workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Tokenize {split_name} DNA",
        )

    if args.max_train_samples is not None:
        tokenized["train"] = tokenized["train"].select(
            range(min(args.max_train_samples, len(tokenized["train"])))
        )
    if args.max_validation_samples is not None and "validation" in tokenized:
        tokenized["validation"] = tokenized["validation"].select(
            range(min(args.max_validation_samples, len(tokenized["validation"])))
        )

    if len(tokenized["train"]) == 0:
        raise ValueError("Training corpus is empty after preprocessing.")
    if args.do_eval and (
        "validation" not in tokenized or len(tokenized["validation"]) == 0
    ):
        raise ValueError("Validation corpus is empty or unavailable after preprocessing.")
    return tokenized


def _validate_model_sequence_length(config: Any, args: argparse.Namespace) -> None:
    """Ensure the model can represent the requested token sequence length."""

    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is not None and int(max_positions) < args.max_seq_length:
        raise ValueError(
            f"Model max_position_embeddings={max_positions} is smaller than "
            f"--max-seq-length={args.max_seq_length}."
        )


def _set_model_special_token_ids(config: Any, tokenizer: Any) -> None:
    """Synchronize model configuration with tokenizer special-token IDs."""

    token_attributes = {
        "pad_token_id": "pad_token_id",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "sep_token_id": "sep_token_id",
    }
    for config_attribute, tokenizer_attribute in token_attributes.items():
        token_id = getattr(tokenizer, tokenizer_attribute, None)
        if token_id is not None:
            setattr(config, config_attribute, int(token_id))


def build_model_config(args: argparse.Namespace, tokenizer: Any) -> Any:
    """Build a configuration for random initialization.

    When --model-path is supplied, only config.json is loaded from that local
    model directory. Existing model weights are deliberately ignored.
    """

    if args.model_path is not None:
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {args.model_path}")
        config = AutoConfig.from_pretrained(
            str(args.model_path),
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.config_name_or_path:
        config = AutoConfig.from_pretrained(
            args.config_name_or_path,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError("--hidden-size must be divisible by --num-attention-heads.")
        max_positions = args.max_position_embeddings or max(512, args.max_seq_length)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=max_positions,
            type_vocab_size=1,
        )

    # Random initialization can safely adapt the embedding/output dimensions to
    # the current tokenizer because no pretrained embedding weights are retained.
    config.vocab_size = len(tokenizer)
    _set_model_special_token_ids(config, tokenizer)
    _validate_model_sequence_length(config, args)
    return config


def load_or_initialize_model(args: argparse.Namespace, tokenizer: Any) -> Any:
    """Create a new model or inherit weights from a complete local model."""

    if args.model_init == "from_pretrained":
        if args.model_path is None:
            raise ValueError("--model-path is required when --model-init=from_pretrained.")
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {args.model_path}")
        if args.config_name_or_path:
            raise ValueError(
                "--config-name-or-path cannot be combined with "
                "--model-init=from_pretrained; the configuration is loaded from --model-path."
            )

        model = AutoModelForMaskedLM.from_pretrained(
            str(args.model_path),
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
        loaded_vocab_size = int(getattr(model.config, "vocab_size", -1))
        tokenizer_vocab_size = len(tokenizer)
        if loaded_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                "Pretrained model/tokenizer vocabulary mismatch: "
                f"model vocab_size={loaded_vocab_size}, tokenizer size={tokenizer_vocab_size}. "
                "Use the tokenizer paired with the local model, or initialize from scratch."
            )
        _set_model_special_token_ids(model.config, tokenizer)
        _validate_model_sequence_length(model.config, args)
        LOGGER.info("Loaded pretrained model weights from local directory: %s", args.model_path)
        return model

    config = build_model_config(args, tokenizer)
    model = AutoModelForMaskedLM.from_config(
        config,
        trust_remote_code=args.trust_remote_code,
    )
    LOGGER.info(
        "Randomly initialized model weights from %s",
        args.model_path if args.model_path is not None else "the effective configuration",
    )
    return model


def add_compatible_warmup_argument(
    values: Dict[str, Any],
    signature: Mapping[str, inspect.Parameter],
    warmup_ratio: float,
) -> None:
    """Set warmup without triggering the Transformers 5.x deprecation warning.

    Transformers 4.x uses ``warmup_ratio`` (normally defaulting to ``0.0``),
    whereas Transformers 5.x marks that argument with a ``None`` default and
    accepts a fractional ratio through ``warmup_steps``. Keep the public CLI
    stable and translate its value according to the installed signature.
    """

    ratio_parameter = signature.get("warmup_ratio")
    if ratio_parameter is not None and ratio_parameter.default is not None:
        values["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in signature:
        values["warmup_steps"] = warmup_ratio
    elif ratio_parameter is not None:
        # Defensive fallback for an unusual signature exposing only the
        # deprecated parameter with a None default.
        values["warmup_ratio"] = warmup_ratio


def make_training_arguments(args: argparse.Namespace, has_eval: bool) -> TrainingArguments:
    """Support both new eval_strategy and legacy evaluation_strategy names."""

    signature = inspect.signature(TrainingArguments.__init__).parameters
    report_to: List[str] = [] if args.report_to == ["none"] else args.report_to
    values: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": args.do_train,
        "do_eval": args.do_eval and has_eval,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "save_strategy": args.save_strategy,
        "logging_strategy": args.logging_strategy,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "full_determinism": args.full_determinism,
        "dataloader_num_workers": args.dataloader_num_workers,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "optim": args.optim,
        "report_to": report_to,
        "run_name": args.run_name,
        "gradient_checkpointing": args.gradient_checkpointing,
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "local_rank": args.local_rank,
        "remove_unused_columns": True,
    }
    add_compatible_warmup_argument(values, signature, args.warmup_ratio)
    values["eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"] = (
        args.eval_strategy if args.do_eval and has_eval else "no"
    )
    if "use_cpu" in signature:
        values["use_cpu"] = args.use_cpu
    elif "no_cuda" in signature:
        values["no_cuda"] = args.use_cpu
    if args.deepspeed:
        values["deepspeed"] = args.deepspeed
    if args.fsdp:
        values["fsdp"] = args.fsdp
    return TrainingArguments(**{key: value for key, value in values.items() if key in signature})


def make_trainer(tokenizer: Any, **kwargs: Any) -> Trainer:
    """Use processing_class on new Transformers and tokenizer on older versions."""

    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**kwargs)
    # Transformers 5.x infers this flag from a model forward(**kwargs)
    # signature. BertForMaskedLM exposes **kwargs but its loss does not consume
    # num_items_in_batch, so Trainer otherwise multiplies mean loss (and its
    # gradients) by the distributed world size. Mark the actual behavior
    # explicitly. The guard remains compatible with older Transformers.
    if hasattr(trainer, "model_accepts_loss_kwargs"):
        trainer.model_accepts_loss_kwargs = False
    return trainer


def resolve_checkpoint(args: argparse.Namespace) -> Optional[str]:
    requested = args.resume_from_checkpoint
    if requested:
        if requested.lower() in {"auto", "last"}:
            checkpoint = get_last_checkpoint(str(args.output_dir)) if args.output_dir.is_dir() else None
            if checkpoint is None:
                LOGGER.info("No existing checkpoint found; starting a new run.")
            return checkpoint
        path = Path(requested)
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
        return str(path)

    if args.do_train and args.output_dir.is_dir() and not args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(str(args.output_dir))
        if checkpoint:
            LOGGER.info("Resuming automatically from %s", checkpoint)
            return checkpoint
        if any(args.output_dir.iterdir()):
            raise ValueError(
                f"Output directory {args.output_dir} is non-empty and has no Trainer checkpoint. "
                "Use --overwrite-output-dir or choose another directory."
            )
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 and --bf16.")
    if args.model_init == "from_scratch" and args.model_path and args.config_name_or_path:
        raise ValueError(
            "Choose only one configuration source in from_scratch mode: "
            "--model-path or --config-name-or-path."
        )
    if not 0.0 < args.mlm_probability < 1.0:
        raise ValueError("--mlm-probability must be between 0 and 1.")
    if args.max_seq_length < 4:
        raise ValueError("--max-seq-length must be at least 4.")

    set_seed(args.seed)
    spec = load_tokenization_spec(args)
    LOGGER.info(
        "DNA tokenization: strategy=%s, k=%d, stride=%d, metadata=%s",
        spec.strategy,
        spec.k,
        spec.stride,
        spec.metadata_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    validate_tokenizer(tokenizer, spec)

    datasets = load_corpus(args, spec, tokenizer)
    model = load_or_initialize_model(args, tokenizer)
    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    has_eval = "validation" in datasets
    training_args = make_training_arguments(args, has_eval)
    collator_kwargs: Dict[str, Any] = {
        "tokenizer": tokenizer,
        "mlm": True,
        "mlm_probability": args.mlm_probability,
        "pad_to_multiple_of": 8 if (args.fp16 or args.bf16) else None,
    }
    # Recent Transformers releases expose an explicit collator seed. Keep
    # compatibility with older releases by checking the constructor signature.
    if "seed" in inspect.signature(DataCollatorForLanguageModeling.__init__).parameters:
        collator_kwargs["seed"] = args.data_seed
    collator = DataCollatorForLanguageModeling(**collator_kwargs)
    trainer = make_trainer(
        tokenizer,
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if args.do_train else None,
        eval_dataset=datasets["validation"] if args.do_eval and has_eval else None,
        data_collator=collator,
    )

    checkpoint = resolve_checkpoint(args)
    if args.do_train:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = dict(result.metrics)
        metrics["train_samples"] = len(datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()
        # Only the global main process writes tokenizer artifacts and custom metadata.
        if trainer.is_world_process_zero():
            save_tokenizer_artifacts(
                args.output_dir,
                tokenizer,
                spec,
                args.max_seq_length,
            )

    if args.do_eval:
        if not has_eval:
            raise ValueError("Evaluation was requested but no validation split is available.")
        metrics = dict(trainer.evaluate())
        metrics["eval_samples"] = len(datasets["validation"])
        loss = metrics.get("eval_loss")
        if loss is not None:
            try:
                metrics["perplexity"] = math.exp(float(loss))
            except OverflowError:
                metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Avoid relying on nondeterministic NCCL teardown at interpreter exit.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

