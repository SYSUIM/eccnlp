export CUDA_VISIBLE_DEVICES="0, 2, 4"
python -u inference.py \
    --bert_tokenizer_path /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_model_path /data2/panziyang/project/bertForSequenceClassification/checkpoint/bert-base-chinese_focalloss_nofilter_over_raw/checkpoint-14000 \
    --bert_batch_size 512 \
    --data /data/zyx2022/FinanceText/process_file/3.1_raw_dataset_dict_nocut.txt\
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
