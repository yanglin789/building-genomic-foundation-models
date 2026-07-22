python ../scripts/fasta_to_atgc.py \
  --input-path data/raw/synthetic.fasta  \
  --output-path data/processed/pretrain_corpus.txt \
  --window-size 510 \
  --stride 510 \
  --ambiguous-strategy split