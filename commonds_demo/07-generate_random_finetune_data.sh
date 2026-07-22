python ../scripts/generate_random_finetune_data.py \
  --train-output data/processed/finetune_train.csv \
  --validation-output data/processed/finetune_validation.csv \
  --test-output data/processed/finetune_test.csv \
  --num-sequences 10000 \
  --sequence-length 200 \
  --positive-fraction 0.50 \
  --motif ATGCGTAC \
  --split-ratio 0.8,0.1,0.1

  