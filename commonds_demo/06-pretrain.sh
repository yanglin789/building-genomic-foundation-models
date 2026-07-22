python ../scripts/pretrain.py \
  --train-file data/processed/pretrain_corpus.txt \
  --tokenizer-path model/genomic_tokenizer \
  --model-init from_scratch \
  --model-path model/genomic_bert_model \
  --output-dir outputs/genomic_bert_pretrained \
  --max-seq-length 512 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 

