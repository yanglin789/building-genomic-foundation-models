python ../scripts/finetune.py \
  --model-name-or-path outputs/genomic_bert_pretrained \
  --train-file data/processed/finetune_train.csv \
  --validation-file data/processed/finetune_validation.csv \
  --test-file data/processed/finetune_test.csv \
  --task-type binary_classification \
  --output-dir outputs/finetune_binary_classification \
  --max-seq-length 512 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 2e-4 \

