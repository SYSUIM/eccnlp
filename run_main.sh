nohup python -u main.py \
    --data /data/zyx2022/FinanceText/process_file/2.1_raw_dataset_dict.txt \
    --balance down \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --num_epochs 20 >>./log/$(date +%Y%m%d).log 2>&1 &