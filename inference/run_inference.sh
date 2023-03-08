# nohup python -u main.py \
python -u inference.py \
    --bert_tokenizer_path /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_model_path /data2/panziyang/project/bertForSequenceClassification/checkpoint/bert-base-chinese_focalloss_refilter_over/checkpoint-8000 \
    --data /data/zyx2022/FinanceText/process_file/2.1_raw_dataset_dict.txt \
    --predict_data /data/xf2022/Projects/eccnlp_local/data/dataset/3.1_data_dict_length5_test.txt \
    --balance down \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --ensemble_date 03.07 \
    --num_epochs 20 \
    --save_dir /data/pzy2022/project/eccnlp/checkpoint/20230228 \
    --device gpu:3 
    # >/dev/null 2>&1 &
    # >>./log/$(date +%Y%m%d).log 2>&1 &