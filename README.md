# Building Genomic Foundation Models: From Pretraining to Fine-Tuning

This repository provides an end-to-end implementation for building genomic foundation models, including genome sequence preprocessing, DNA tokenization, masked language model (MLM) pretraining, supervised fine-tuning, and model evaluation.

The workflow is primarily implemented using the Hugging Face framework, including Transformers, Tokenizers, Datasets, and Evaluate, and is designed to facilitate reproducible genomic language model development and adaptation to different genomes, tokenization strategies, Transformer architectures, and downstream genomic applications.


## Features

- Generate deterministic synthetic FASTA data with configurable GC and `N` content.
- Convert FASTA records into fixed-length `A/T/G/C` training samples.
- Build and validate single-nucleotide or k-mer Hugging Face tokenizers.
- Initialize and pretrain a BERT masked language model from scratch.
- Fine-tune models for binary, multiclass, or multilabel classification, or regression.
- Run on CPU, one GPU, or multiple GPUs with `torchrun`.

## Repository layout

```text
.
├── commonds_demo/   # Numbered end-to-end shell examples
└── scripts/         # Python command-line tools
```

## Quick start

Run the examples from `commonds_demo` so their relative paths resolve correctly:

```bash
cd commonds_demo
bash 01-generate_random_fasta_data.sh
bash 02-fasta_to_atgc.sh
bash 03-build_tokenizer.sh
bash 04-tokenizer_validate.sh
bash 05-build_model_config.sh
bash 06-pretrain.sh
bash 07-generate_random_finetune_data.sh
bash 08-finetune.sh
bash 09-inference.sh
```

The default example performs the following workflow:

1. Generates a reproducible synthetic genome.
2. Creates a plain-text pretraining corpus with one 510-base DNA sample per line.
3. Builds and validates a single-nucleotide tokenizer. Special tokens expand each sample to 512 tokens.
4. Initializes a BERT-base masked language model.
5. Pretrains the model and saves checkpoints and metrics.
6. Generates motif-based binary-classification CSV files.
7. Fine-tunes the pretrained model and evaluates it on validation and test data.
