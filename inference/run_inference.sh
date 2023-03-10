# nohup python -u main.py \
nohup python -u inference.py \
    --bert_tokenizer_path /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_model_path /data2/panziyang/project/bertForSequenceClassification/checkpoint/bert-base-chinese_focalloss_refilter_over/checkpoint-8000 \
    --data /data/fkj2023/Project/eccnlp_local/data_process/3.1_data_dict_length0_test.txt \
    --predict_data /data/xf2022/Projects/eccnlp_local/data/dataset/3.1_data_dict_length5_test.txt \
    --balance down \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --ensemble_date 03.07 \
    --num_epochs 10 \
    --save_dir /data/pzy2022/project/eccnlp/checkpoint/20230228 \
    --device gpu:0 \
    >>./log/$(date +%Y%m%d).log 2>&1 &
    # >/dev/null 2>&1 &
    # --data /data/zyx2022/FinanceText/process_file/3.1_data_dict_length0.txt \ 
