python finetune.py \
    --train_path ./data/train_validation_test/dataset1/train_data.txt \
    --dev_path ./data/train_validation_test/dataset1/validation_data.txt \
    --save_dir ./checkpoint/checkpoint3_20221017 \
    --learning_rate 1e-6 \
    --batch_size 8 \
    --max_seq_len 512 \
    --num_epochs 50 \
    --model uie-base \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu


