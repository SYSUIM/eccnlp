nohup python -u main.py \
    --data ./data/raw_data/第一第二期手工标注数据汇总-20221214.xlsx \
    --balance down \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --num_epochs 20