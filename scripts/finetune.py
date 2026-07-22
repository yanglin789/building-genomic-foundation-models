#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch.distributed as dist

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger("dna_finetune")
DNA_METADATA_NAME = "dna_tokenizer_config.json"
REQUIRED_SPECIAL_TOKENS = ("pad_token", "unk_token", "cls_token", "sep_token")
SEQUENCE_COLUMN = "sequence"
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class DnaTokenizationSpec:
    strategy: str
    k: int
    stride: int
    alphabet: str
    ambiguous_policy: str
    case_policy: str
    whitespace_policy: str
    metadata_path: Optional[Path]
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelSchema:
    task_type: str
    num_labels: int
    label_names: Tuple[str, ...]
    multilabel_is_vector: bool = False

    @property
    def label2id(self) -> Dict[str, int]:
        return {name: index for index, name in enumerate(self.label_names)}

    @property
    def id2label(self) -> Dict[int, str]:
        return {index: name for index, name in enumerate(self.label_names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained DNA model.")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="训练集 CSV；表头必须为 sequence,label。",
    )
    data.add_argument(
        "--validation-file",
        type=Path,
        help="可选验证集 CSV；表头必须为 sequence,label。",
    )
    data.add_argument(
        "--test-file",
        type=Path,
        help="可选测试集 CSV；表头必须为 sequence,label。",
    )
    data.add_argument("--dataset-cache-dir", type=Path)
    data.add_argument("--preprocessing-num-workers", type=int, default=1)
    data.add_argument("--overwrite-cache", action="store_true")
    data.add_argument("--max-train-samples", type=int)
    data.add_argument("--max-validation-samples", type=int)
    data.add_argument("--max-test-samples", type=int)

    task = parser.add_argument_group("task")
    task.add_argument(
        "--task-type",
        choices=(
            "binary_classification",
            "multiclass_classification",
            "regression",
            "multilabel_classification",
        ),
        default="binary_classification",
        help=(
            "任务类型：二分类固定 2 类，多分类至少 3 类，"
            "回归固定一个连续目标，多标签分类至少 2 个标签。"
        ),
    )
    task.add_argument(
        "--label-list",
        nargs="+",
        help="显式标签顺序；不提供时从训练集推断。可使用空格或单个逗号分隔字符串。",
    )
    task.add_argument(
        "--num-labels",
        type=int,
        help="与数据标签数进行一致性检查；回归任务固定为 1。",
    )
    task.add_argument("--label-delimiter", default=",", help="CSV 多标签字符串的分隔符。")
    task.add_argument(
        "--multilabel-input-format",
        choices=("auto", "labels", "multi_hot"),
        default="auto",
        help="多标签列是标签集合还是 0/1 向量。",
    )
    task.add_argument("--multilabel-threshold", type=float, default=0.5)

    tok = parser.add_argument_group("tokenizer")
    tok.add_argument("--tokenizer-path", help="默认与 --model-name-or-path 相同。")
    tok.add_argument("--tokenizer-metadata", type=Path)
    tok.add_argument("--tokenization-strategy", choices=("auto", "single", "kmer"), default="auto")
    tok.add_argument("--kmer-size", type=int)
    tok.add_argument("--kmer-stride", type=int)
    tok.add_argument("--ambiguous-policy", choices=("metadata", "error", "clean"), default="metadata")
    tok.add_argument("--max-seq-length", type=int, default=512)

    model = parser.add_argument_group("model")
    model.add_argument("--model-name-or-path", required=True, help="预训练模型目录或 Hub 模型名。")
    model.add_argument("--model-revision", default="main")
    model.add_argument("--cache-dir", type=Path)
    model.add_argument("--trust-remote-code", action="store_true")
    model.add_argument(
        "--strict-head-shape",
        action="store_true",
        help="分类头尺寸不匹配时抛错；默认重新初始化任务头。",
    )
    model.add_argument("--gradient-checkpointing", action="store_true")

    train = parser.add_argument_group("training")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--overwrite-output-dir", action="store_true")
    train.add_argument("--resume-from-checkpoint", help="检查点目录，或 auto/last。")
    train.add_argument("--no-train", dest="do_train", action="store_false")
    train.set_defaults(do_train=True)
    train.add_argument("--do-eval", dest="do_eval", action="store_true")
    train.add_argument("--no-eval", dest="do_eval", action="store_false")
    train.add_argument("--do-predict", dest="do_predict", action="store_true")
    train.add_argument("--no-predict", dest="do_predict", action="store_false")
    train.set_defaults(do_eval=None, do_predict=None)
    train.add_argument("--num-train-epochs", type=float, default=3.0)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--per-device-train-batch-size", type=int, default=16)
    train.add_argument("--per-device-eval-batch-size", type=int, default=32)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--lr-scheduler-type", default="linear")
    train.add_argument("--eval-strategy", choices=("no", "steps", "epoch"), default="epoch")
    train.add_argument("--save-strategy", choices=("no", "steps", "epoch"), default="epoch")
    train.add_argument("--logging-strategy", choices=("no", "steps", "epoch"), default="steps")
    train.add_argument("--logging-steps", type=int, default=25)
    train.add_argument("--eval-steps", type=int)
    train.add_argument("--save-steps", type=int, default=500)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--load-best-model-at-end", action="store_true")
    train.add_argument("--metric-for-best-model")
    train.add_argument("--greater-is-better", dest="greater_is_better", action="store_true")
    train.add_argument("--lower-is-better", dest="greater_is_better", action="store_false")
    train.set_defaults(greater_is_better=None)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--data-seed", type=int, default=42)
    train.add_argument(
        "--full-determinism",
        action="store_true",
        help="启用 Transformers/PyTorch 的完全确定性算法（可能降低训练速度）。",
    )
    train.add_argument("--dataloader-num-workers", type=int, default=0)
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--tf32", action="store_true")
    train.add_argument("--use-cpu", action="store_true")
    train.add_argument("--optim", default="adamw_torch")
    train.add_argument("--report-to", nargs="*", default=["none"])
    train.add_argument("--run-name")
    train.add_argument("--deepspeed")
    train.add_argument("--fsdp")
    train.add_argument("--ddp-find-unused-parameters", action="store_true")
    train.add_argument("--local-rank", "--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_tokenization_spec(args: argparse.Namespace, tokenizer_ref: str) -> DnaTokenizationSpec:
    metadata_path = args.tokenizer_metadata or (Path(tokenizer_ref) / DNA_METADATA_NAME)
    metadata: Dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"DNA tokenizer metadata must be a JSON object: {metadata_path}")
    elif args.tokenization_strategy == "auto":
        raise FileNotFoundError(
            f"Cannot find {metadata_path}. Pass --tokenizer-metadata or "
            "an explicit --tokenization-strategy."
        )
    else:
        metadata_path = None

    strategy = (
        str(metadata.get("tokenization_strategy", "single"))
        if args.tokenization_strategy == "auto"
        else args.tokenization_strategy
    ).lower()
    if strategy not in {"single", "kmer"}:
        raise ValueError(f"Unsupported tokenization strategy: {strategy!r}")
    if strategy == "single":
        k, stride = 1, 1
    else:
        k = int(args.kmer_size if args.kmer_size is not None else metadata.get("k", 6))
        stride = int(
            args.kmer_stride
            if args.kmer_stride is not None
            else metadata.get("stride", k)
        )
    if k < 1 or stride < 1 or stride > k:
        raise ValueError(f"Expected 1 <= stride <= k, got k={k}, stride={stride}.")
    policy = str(metadata.get("ambiguous_policy", "error")).lower()
    if args.ambiguous_policy != "metadata":
        policy = args.ambiguous_policy
    if policy not in {"error", "clean"}:
        raise ValueError(f"Unsupported ambiguous_policy: {policy!r}")
    return DnaTokenizationSpec(
        strategy=strategy,
        k=k,
        stride=stride,
        alphabet=str(metadata.get("alphabet", "ATGC")).upper(),
        ambiguous_policy=policy,
        case_policy=str(metadata.get("case_policy", "upper")).lower(),
        whitespace_policy=str(metadata.get("whitespace_policy", "remove")).lower(),
        metadata_path=metadata_path,
        source_metadata=dict(metadata),
    )


def validate_tokenizer_compatibility(tokenizer: Any, spec: DnaTokenizationSpec) -> None:
    """Reject tokenizer/metadata pairs that cannot represent the declared DNA vocabulary."""
    metadata = spec.source_metadata
    if "vocab_size" in metadata:
        try:
            declared_vocab_size = int(metadata["vocab_size"])
        except (TypeError, ValueError) as exc:
            raise ValueError("DNA tokenizer metadata vocab_size must be an integer.") from exc
        if declared_vocab_size != len(tokenizer):
            raise ValueError(
                f"Tokenizer vocabulary size {len(tokenizer)} does not match "
                f"metadata vocab_size={declared_vocab_size}."
            )

    declared_specials = metadata.get("special_tokens", {})
    if "special_tokens" in metadata and not isinstance(declared_specials, Mapping):
        raise ValueError("DNA tokenizer metadata special_tokens must be a JSON object.")
    if "special_tokens" in metadata:
        missing = [name for name in REQUIRED_SPECIAL_TOKENS if name not in declared_specials]
        if missing:
            raise ValueError(f"DNA tokenizer metadata is missing special tokens: {missing}")

    vocabulary = tokenizer.get_vocab()
    for name in REQUIRED_SPECIAL_TOKENS:
        token = getattr(tokenizer, name, None)
        token_id = getattr(tokenizer, f"{name}_id", None)
        if token is None or token_id is None:
            raise ValueError(f"The tokenizer must define {name}.")
        mapped_id = tokenizer.convert_tokens_to_ids(token)
        recovered_token = tokenizer.convert_ids_to_tokens(token_id)
        if mapped_id != token_id or vocabulary.get(str(token)) != token_id:
            raise ValueError(f"Tokenizer mapping for {name}={token!r} is inconsistent.")
        if str(recovered_token) != str(token):
            raise ValueError(f"Tokenizer reverse mapping for {name}={token!r} is inconsistent.")
        declared = declared_specials.get(name, metadata.get(name))
        if isinstance(declared, Mapping):
            declared = declared.get("content")
        if declared is not None and str(declared) != str(token):
            raise ValueError(
                f"Metadata {name}={declared!r} does not match tokenizer value {token!r}."
            )


def build_output_tokenizer_metadata(
    tokenizer: Any,
    spec: DnaTokenizationSpec,
    max_seq_length: int,
) -> Dict[str, Any]:
    """Preserve source metadata while recording the effective fine-tuning tokenizer."""
    metadata = dict(spec.source_metadata)
    existing_specials = metadata.get("special_tokens")
    special_tokens = dict(existing_specials) if isinstance(existing_specials, Mapping) else {}
    for name in (*REQUIRED_SPECIAL_TOKENS, "mask_token"):
        token = getattr(tokenizer, name, None)
        if token is not None:
            special_tokens[name] = str(token)
    metadata.update(
        {
            "format_version": 1,
            "tokenization_strategy": spec.strategy,
            "k": spec.k,
            "stride": spec.stride,
            "alphabet": spec.alphabet,
            "ambiguous_policy": spec.ambiguous_policy,
            "case_policy": spec.case_policy,
            "whitespace_policy": spec.whitespace_policy,
            "vocab_size": len(tokenizer),
            "model_max_length": max_seq_length,
            "requires_pretokenization": True,
            "special_tokens": special_tokens,
        }
    )
    return metadata


def save_tokenizer_with_metadata(
    output_dir: Path,
    tokenizer: Any,
    spec: DnaTokenizationSpec,
    max_seq_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.model_max_length = max_seq_length
    tokenizer.save_pretrained(str(output_dir))
    metadata = build_output_tokenizer_metadata(tokenizer, spec, max_seq_length)
    target = output_dir / DNA_METADATA_NAME
    target.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_dna(value: Any, spec: DnaTokenizationSpec) -> str:
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
    if spec.strategy == "single":
        return list(sequence)
    return [sequence[start : start + spec.k] for start in range(0, len(sequence), spec.stride)]


def validate_csv_file(path: Path, split: str) -> None:
    """严格检查微调 CSV 的文件类型、表头和每一行的基本完整性。"""

    if not path.is_file():
        raise FileNotFoundError(f"{split} CSV does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{split} dataset must be a .csv file: {path}")

    required_columns = [SEQUENCE_COLUMN, LABEL_COLUMN]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{split} CSV has no header: {path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{split} CSV contains duplicate column names: {fieldnames}")
        if fieldnames != required_columns:
            raise ValueError(
                f"{split} CSV header must be exactly {required_columns}, got {fieldnames}."
            )

        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if None in row:
                raise ValueError(
                    f"{split} CSV line {line_number} has extra unquoted fields. "
                    "Multi-label values containing commas must be enclosed in double quotes, "
                    'for example "1,0,1,0".'
                )
            sequence = row.get(SEQUENCE_COLUMN)
            label = row.get(LABEL_COLUMN)
            if sequence is None or not str(sequence).strip():
                raise ValueError(f"{split} CSV line {line_number} has an empty sequence.")
            if label is None or not str(label).strip():
                raise ValueError(f"{split} CSV line {line_number} has an empty label.")

    if row_count == 0:
        raise ValueError(f"{split} CSV contains no data rows: {path}")


def load_raw_datasets(args: argparse.Namespace) -> DatasetDict:
    """加载标准 CSV 数据；每个文件必须且只能包含 sequence、label 两列。"""

    paths = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file,
    }
    data_files: Dict[str, str] = {}
    for split, path in paths.items():
        if path is None:
            continue
        validate_csv_file(path, split)
        data_files[split] = str(path)

    raw = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    if not isinstance(raw, DatasetDict):
        raw = DatasetDict(raw)

    expected_columns = [SEQUENCE_COLUMN, LABEL_COLUMN]
    for split, dataset in raw.items():
        if dataset.column_names != expected_columns:
            raise ValueError(
                f"{split} dataset columns changed during loading: "
                f"expected {expected_columns}, got {dataset.column_names}."
            )
    return raw

def parse_label_list_argument(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    if len(values) == 1 and "," in values[0]:
        values = values[0].split(",")
    labels = [str(value).strip() for value in values if str(value).strip()]
    if len(labels) != len(set(labels)):
        raise ValueError("--label-list contains duplicate labels.")
    return labels


def parse_multilabel_value(value: Any, delimiter: str) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected a JSON list for multilabel data, got: {value!r}")
            return parsed
        return [item.strip() for item in stripped.split(delimiter) if item.strip()]
    return [value]


def is_binary(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) in {0.0, 1.0}
    return False


def _label_sort_key(value: str) -> Tuple[Any, ...]:
    """Sort numeric labels numerically while keeping arbitrary text labels deterministic."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return (1, str(value))
    if math.isfinite(numeric):
        return (0, numeric, str(value))
    return (1, str(value))


def infer_label_schema(args: argparse.Namespace, train_dataset: Dataset) -> LabelSchema:
    """Infer and validate the task head exclusively from the training split."""

    explicit_labels = parse_label_list_argument(args.label_list)
    raw_labels = train_dataset[LABEL_COLUMN]
    if not raw_labels:
        raise ValueError("Cannot infer labels from an empty training dataset.")

    if args.task_type == "regression":
        if explicit_labels is not None:
            raise ValueError("Regression does not accept --label-list.")
        if args.num_labels not in {None, 1}:
            raise ValueError("Regression requires --num-labels 1.")
        for value in raw_labels:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Regression label is not numeric: {value!r}.") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"Regression labels must be finite, got {value!r}.")
        return LabelSchema("regression", 1, ("score",))

    if args.task_type in {"binary_classification", "multiclass_classification"}:
        labels = explicit_labels or sorted(
            {str(value) for value in raw_labels}, key=_label_sort_key
        )
        if args.task_type == "binary_classification" and len(labels) != 2:
            raise ValueError(
                "binary_classification requires exactly 2 labels; "
                f"found {len(labels)} ({labels})."
            )
        if args.task_type == "multiclass_classification" and len(labels) < 3:
            raise ValueError(
                "multiclass_classification requires at least 3 labels; "
                f"found {len(labels)} ({labels})."
            )
        if args.num_labels is not None and args.num_labels != len(labels):
            raise ValueError(
                f"--num-labels={args.num_labels} but the label vocabulary has "
                f"{len(labels)} entries."
            )
        return LabelSchema(args.task_type, len(labels), tuple(labels))

    # Multi-label CSV generated by this project contains fixed-width multi-hot
    # vectors. ``labels`` remains available for external datasets that store a
    # list of active label names instead.
    parsed = [parse_multilabel_value(value, args.label_delimiter) for value in raw_labels]
    same_length = bool(parsed) and len({len(items) for items in parsed}) == 1
    looks_multi_hot = same_length and len(parsed[0]) > 0 and all(
        is_binary(item) for items in parsed for item in items
    )
    if args.multilabel_input_format == "multi_hot":
        is_vector = True
    elif args.multilabel_input_format == "labels":
        is_vector = False
    else:
        is_vector = looks_multi_hot

    if is_vector:
        if not same_length or not all(is_binary(item) for items in parsed for item in items):
            raise ValueError(
                "multi_hot labels must be equal-length vectors containing only 0/1 values."
            )
        width = len(parsed[0])
        labels = explicit_labels or [str(index) for index in range(width)]
        if len(labels) != width:
            raise ValueError(
                f"--label-list has {len(labels)} entries but multi-hot vectors have width {width}."
            )
    else:
        labels = explicit_labels or sorted(
            {str(item) for items in parsed for item in items}, key=_label_sort_key
        )
    if len(labels) < 2:
        raise ValueError(
            "multilabel_classification requires at least 2 labels; "
            f"found {len(labels)} ({labels})."
        )
    if args.num_labels is not None and args.num_labels != len(labels):
        raise ValueError(f"--num-labels={args.num_labels} but inferred {len(labels)} labels.")
    return LabelSchema(
        "multilabel_classification",
        len(labels),
        tuple(labels),
        multilabel_is_vector=is_vector,
    )


def encode_label(value: Any, schema: LabelSchema, args: argparse.Namespace) -> Any:
    """Convert one raw target to the exact representation expected by Transformers."""

    if schema.task_type == "regression":
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Regression label is not numeric: {value!r}.") from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"Regression labels must be finite, got {value!r}.")
        return numeric_value
    if schema.task_type in {"binary_classification", "multiclass_classification"}:
        key = str(value)
        if key not in schema.label2id:
            raise ValueError(f"Unknown classification label: {value!r}")
        return schema.label2id[key]

    items = parse_multilabel_value(value, args.label_delimiter)
    if schema.multilabel_is_vector:
        if len(items) != schema.num_labels or not all(is_binary(item) for item in items):
            raise ValueError(f"Expected a {schema.num_labels}-element binary multi-hot vector.")
        return [float(item) for item in items]
    vector = [0.0] * schema.num_labels
    for item in items:
        key = str(item)
        if key not in schema.label2id:
            raise ValueError(f"Unknown multilabel class: {item!r}")
        vector[schema.label2id[key]] = 1.0
    return vector

def preprocess_datasets(
    args: argparse.Namespace,
    raw: DatasetDict,
    tokenizer: Any,
    spec: DnaTokenizationSpec,
    schema: LabelSchema,
) -> DatasetDict:
    workers = args.preprocessing_num_workers if args.preprocessing_num_workers > 1 else None

    def preprocess_batch(examples: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
        sequences = [normalize_dna(value, spec) for value in examples[SEQUENCE_COLUMN]]
        if any(not sequence for sequence in sequences):
            raise ValueError("A DNA sequence became empty after normalization.")
        token_lists = [dna_to_tokens(sequence, spec) for sequence in sequences]
        encoded = tokenizer(
            token_lists,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        if any(
            tokenizer.unk_token_id in input_ids
            for input_ids in encoded["input_ids"]
        ):
            raise ValueError(
                "DNA tokenization produced [UNK]; the tokenizer and DNA metadata "
                "are incompatible with the selected tokenization specification."
            )
        encoded["labels"] = [
            encode_label(value, schema, args) for value in examples[LABEL_COLUMN]
        ]
        return encoded

    result = DatasetDict()
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
        "test": args.max_test_samples,
    }
    for split, dataset in raw.items():
        if limits[split] is not None:
            dataset = dataset.select(range(min(limits[split], len(dataset))))
        result[split] = dataset.map(
            preprocess_batch,
            batched=True,
            num_proc=workers,
            remove_columns=dataset.column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Tokenize {split} DNA",
        )
        if len(result[split]) == 0:
            raise ValueError(f"{split} dataset is empty.")
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero for an undefined ratio instead of emitting NaN or warnings."""

    return float(numerator / denominator) if denominator else 0.0


def _precision_recall_f1(
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> Tuple[float, float, float]:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _binary_roc_auc(y_true: np.ndarray, positive_scores: np.ndarray) -> float:
    """Compute tie-aware ROC-AUC with NumPy using the Mann-Whitney statistic.

    ROC-AUC is mathematically undefined when a split contains only one class.
    Returning 0.0 follows this module's zero-denominator policy and, importantly,
    keeps evaluation/checkpoint selection deterministic instead of raising.
    """

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if truth.size != scores.size or truth.size == 0:
        raise ValueError("ROC-AUC requires equally sized, non-empty targets and scores.")
    positive_count = int(np.sum(truth == 1))
    negative_count = int(np.sum(truth == 0))
    if positive_count == 0 or negative_count == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; ties receive their average rank.
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_positive = float(np.sum(ranks[truth == 1]))
    auc = (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(np.clip(auc, 0.0, 1.0))


def _single_label_targets(labels: np.ndarray, num_labels: int) -> np.ndarray:
    raw = np.asarray(labels).reshape(-1)
    if raw.size == 0 or not np.all(np.isfinite(raw.astype(np.float64))):
        raise ValueError("Classification metrics require non-empty, finite label ids.")
    truth = raw.astype(np.int64)
    if not np.all(raw == truth) or np.any(truth < 0) or np.any(truth >= num_labels):
        raise ValueError(f"Classification label ids must be integers in [0, {num_labels - 1}].")
    return truth


def _macro_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_labels: int,
) -> Tuple[float, float, float]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    for label_id in range(num_labels):
        true_positive = float(np.sum((y_true == label_id) & (y_pred == label_id)))
        false_positive = float(np.sum((y_true != label_id) & (y_pred == label_id)))
        false_negative = float(np.sum((y_true == label_id) & (y_pred != label_id)))
        precision, recall, f1 = _precision_recall_f1(
            true_positive, false_positive, false_negative
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1_scores)),
    )


def build_compute_metrics(schema: LabelSchema, threshold: float):
    """Build a dependency-free NumPy metric function for Hugging Face Trainer."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("The multilabel threshold must be strictly between 0 and 1.")

    def compute_metrics(prediction: EvalPrediction) -> Dict[str, float]:
        raw_predictions = prediction.predictions
        logits = np.asarray(
            raw_predictions[0] if isinstance(raw_predictions, tuple) else raw_predictions
        )
        labels = np.asarray(prediction.label_ids)

        if schema.task_type == "regression":
            predicted = logits.astype(np.float64).reshape(-1)
            truth = labels.astype(np.float64).reshape(-1)
            if predicted.size != truth.size or truth.size == 0:
                raise ValueError("Regression metrics require equally sized, non-empty arrays.")
            errors = predicted - truth
            mse = float(np.mean(np.square(errors)))
            mae = float(np.mean(np.abs(errors)))
            centered_truth = truth - np.mean(truth)
            centered_prediction = predicted - np.mean(predicted)
            total_sum_squares = float(np.sum(np.square(centered_truth)))
            r2 = 1.0 - _safe_divide(float(np.sum(np.square(errors))), total_sum_squares)
            if total_sum_squares == 0.0:
                r2 = 0.0
            pearson_denominator = math.sqrt(
                float(np.sum(np.square(centered_truth)))
                * float(np.sum(np.square(centered_prediction)))
            )
            pearson = _safe_divide(
                float(np.sum(centered_truth * centered_prediction)), pearson_denominator
            )
            return {
                "mse": mse,
                "rmse": float(math.sqrt(mse)),
                "mae": mae,
                "r2": float(r2),
                "pearson": pearson,
            }

        if schema.task_type in {"binary_classification", "multiclass_classification"}:
            if logits.ndim != 2 or logits.shape[1] != schema.num_labels:
                raise ValueError(
                    f"Expected logits shape (N, {schema.num_labels}), got {logits.shape}."
                )
            truth = _single_label_targets(labels, schema.num_labels)
            predicted = np.argmax(logits, axis=1).astype(np.int64)
            if predicted.size != truth.size:
                raise ValueError("Prediction and target counts do not match.")
            accuracy = float(np.mean(predicted == truth))

            if schema.task_type == "binary_classification":
                true_positive = float(np.sum((truth == 1) & (predicted == 1)))
                true_negative = float(np.sum((truth == 0) & (predicted == 0)))
                false_positive = float(np.sum((truth == 0) & (predicted == 1)))
                false_negative = float(np.sum((truth == 1) & (predicted == 0)))
                precision, recall, f1 = _precision_recall_f1(
                    true_positive, false_positive, false_negative
                )
                mcc_denominator = math.sqrt(
                    (true_positive + false_positive)
                    * (true_positive + false_negative)
                    * (true_negative + false_positive)
                    * (true_negative + false_negative)
                )
                mcc = _safe_divide(
                    true_positive * true_negative - false_positive * false_negative,
                    mcc_denominator,
                )
                logit_difference = np.clip(
                    logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64),
                    -50.0,
                    50.0,
                )
                positive_probabilities = 1.0 / (1.0 + np.exp(-logit_difference))
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "roc_auc": _binary_roc_auc(truth, positive_probabilities),
                }

            macro_precision, macro_recall, macro_f1 = _macro_classification_metrics(
                truth, predicted, schema.num_labels
            )
            return {
                "accuracy": accuracy,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }

        if logits.ndim != 2 or logits.shape[1] != schema.num_labels:
            raise ValueError(
                f"Expected multilabel logits shape (N, {schema.num_labels}), got {logits.shape}."
            )
        truth_raw = labels.astype(np.float64)
        if truth_raw.shape != logits.shape or not np.all(np.isin(truth_raw, (0.0, 1.0))):
            raise ValueError("Multilabel targets must be a binary matrix matching logits shape.")
        truth = truth_raw.astype(np.int64)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        predicted = (probabilities >= threshold).astype(np.int64)

        micro_tp = float(np.sum((truth == 1) & (predicted == 1)))
        micro_fp = float(np.sum((truth == 0) & (predicted == 1)))
        micro_fn = float(np.sum((truth == 1) & (predicted == 0)))
        micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
            micro_tp, micro_fp, micro_fn
        )
        # Average per-label metrics without weighting labels by prevalence.
        per_label_precision: List[float] = []
        per_label_recall: List[float] = []
        per_label_f1: List[float] = []
        for column in range(schema.num_labels):
            label_truth = truth[:, column]
            label_prediction = predicted[:, column]
            tp = float(np.sum((label_truth == 1) & (label_prediction == 1)))
            fp = float(np.sum((label_truth == 0) & (label_prediction == 1)))
            fn = float(np.sum((label_truth == 1) & (label_prediction == 0)))
            precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
            per_label_precision.append(precision)
            per_label_recall.append(recall)
            per_label_f1.append(f1)
        macro_precision = float(np.mean(per_label_precision))
        macro_recall = float(np.mean(per_label_recall))
        macro_f1 = float(np.mean(per_label_f1))
        return {
            "subset_accuracy": float(np.mean(np.all(predicted == truth, axis=1))),
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    return compute_metrics


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


def make_training_arguments(args: argparse.Namespace, schema: LabelSchema) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__).parameters
    metric = args.metric_for_best_model
    if metric is None:
        metric = {
            "binary_classification": "f1",
            "multiclass_classification": "macro_f1",
            "regression": "rmse",
            "multilabel_classification": "macro_f1",
        }[schema.task_type]
    greater = args.greater_is_better
    if greater is None:
        greater = metric not in {
            "loss",
            "eval_loss",
            "mse",
            "eval_mse",
            "rmse",
            "eval_rmse",
            "mae",
            "eval_mae",
        }
    report_to: List[str] = [] if args.report_to == ["none"] else args.report_to
    eval_strategy = args.eval_strategy if args.do_eval else "no"
    if args.load_best_model_at_end and eval_strategy != args.save_strategy:
        raise ValueError("--load-best-model-at-end requires matching --eval-strategy and --save-strategy.")
    values: Dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "do_predict": args.do_predict,
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
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": metric,
        "greater_is_better": greater,
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
    values["eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"] = eval_strategy
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
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**kwargs)
    # Transformers 5.x can infer loss-kwarg support from forward(**kwargs)
    # even when a sequence-classification model does not consume
    # num_items_in_batch. That would scale distributed loss and gradients by
    # world size and break gradient-accumulation normalization. State the
    # model's actual mean-loss behavior explicitly; guard older releases.
    if hasattr(trainer, "model_accepts_loss_kwargs"):
        trainer.model_accepts_loss_kwargs = False
    return trainer


def resolve_checkpoint(args: argparse.Namespace) -> Optional[str]:
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() in {"auto", "last"}:
            return get_last_checkpoint(str(args.output_dir)) if args.output_dir.is_dir() else None
        checkpoint = Path(args.resume_from_checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
        return str(checkpoint)
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
    args.do_eval = args.validation_file is not None if args.do_eval is None else args.do_eval
    args.do_predict = args.test_file is not None if args.do_predict is None else args.do_predict
    if args.do_eval and args.validation_file is None:
        raise ValueError("--do-eval requires --validation-file.")
    if args.do_predict and args.test_file is None:
        raise ValueError("--do-predict requires --test-file.")
    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 and --bf16.")
    if not 0.0 < args.multilabel_threshold < 1.0:
        raise ValueError("--multilabel-threshold must be between 0 and 1.")
    if args.max_seq_length < 4:
        raise ValueError("--max-seq-length must be at least 4.")
    set_seed(args.seed)

    tokenizer_ref = args.tokenizer_path or args.model_name_or_path
    spec = load_tokenization_spec(args, tokenizer_ref)
    LOGGER.info("DNA tokenization: strategy=%s, k=%d, stride=%d", spec.strategy, spec.k, spec.stride)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    validate_tokenizer_compatibility(tokenizer, spec)

    raw = load_raw_datasets(args)
    LOGGER.info("Loaded CSV splits: %s", {name: len(ds) for name, ds in raw.items()})
    schema = infer_label_schema(args, raw["train"])
    LOGGER.info("Task=%s, labels=%s", schema.task_type, list(schema.label_names))
    datasets = preprocess_datasets(args, raw, tokenizer, spec, schema)

    problem_type = {
        "binary_classification": "single_label_classification",
        "multiclass_classification": "single_label_classification",
        "regression": "regression",
        "multilabel_classification": "multi_label_classification",
    }[schema.task_type]
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        trust_remote_code=args.trust_remote_code,
        num_labels=schema.num_labels,
        problem_type=problem_type,
        label2id=schema.label2id,
        id2label=schema.id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        revision=args.model_revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=not args.strict_head_shape,
    )
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "num_embeddings"):
        raise ValueError("The model must expose input embeddings with num_embeddings.")
    if input_embeddings.num_embeddings != len(tokenizer):
        raise ValueError(
            f"Model input embedding size {input_embeddings.num_embeddings} does not match "
            f"tokenizer vocabulary size {len(tokenizer)}."
        )
    if getattr(model.config, "max_position_embeddings", args.max_seq_length) < args.max_seq_length:
        raise ValueError(
            f"Model max_position_embeddings={model.config.max_position_embeddings} is smaller than "
            f"--max-seq-length={args.max_seq_length}."
        )
    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    training_args = make_training_arguments(args, schema)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )
    trainer = make_trainer(
        tokenizer,
        model=model,
        args=training_args,
        train_dataset=datasets.get("train") if args.do_train else None,
        eval_dataset=datasets.get("validation") if args.do_eval else None,
        data_collator=collator,
        compute_metrics=build_compute_metrics(schema, args.multilabel_threshold),
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
        if trainer.is_world_process_zero():
            save_tokenizer_with_metadata(
                args.output_dir,
                tokenizer,
                spec,
                args.max_seq_length,
            )

    if args.do_eval:
        metrics = dict(trainer.evaluate(eval_dataset=datasets["validation"]))
        metrics["eval_samples"] = len(datasets["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_predict:
        prediction = trainer.predict(datasets["test"], metric_key_prefix="test")
        metrics = dict(prediction.metrics)
        metrics["test_samples"] = len(datasets["test"])
        # Trainer.save_metrics 会写入 output_dir/test_results.json。
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Avoid relying on nondeterministic NCCL teardown at interpreter exit.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


