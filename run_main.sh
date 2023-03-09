nohup python -u main.py \
    --data /data/fkj2023/Project/eccnlp_local/data/2.1_raw_dataset_dict_test.txt \
    --balance down \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --num_epochs 10 \
    --batch_size 64 \
    --UIE_batch_size 4 \
    --UIE_num_epochs 2 \
    --device gpu:2 \
    >>./log/$(date +%Y%m%d).log 2>&1 &