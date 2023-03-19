export CUDA_VISIBLE_DEVICES="2,3"
nohup python -u ./train/train.py \
    --data /data/fkj2023/practice/eccnlp_data/2023-03-18_2.2_raw_dataset_dict_nocut_only_one_text.txt \
    --bert_pretrained_model /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_batch_size 16 \
    --bert_disable_tqdm False \
    --bert_save_dir /data/fkj2023/Project/eccnlp/checkpoint/bert_chinese/$(date +%Y%m%d) \
    --balance up \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --num_epochs 1 \
    --batch_size 16 \
    --UIE_batch_size 4 \
    --UIE_num_epochs 1 \
    --rerank_epoch 30\
    --device gpu \
    --save_dir /data/fkj2023/Project/eccnlp/checkpoint/$(date +%Y%m%d) \
    >>./log/$(date +%Y%m%d)_2.2_nocut_one__train_test2.log 2>&1 &
    # >/dev/null 2>&1 &
    # >>./log/$(date +%Y%m%d).log 2>&1 &
    # >>./log/$(date +%Y%m%d)_2.1_nocut_train_test.log 2>&1 &
    #  /data/fkj2023/practice/eccnlp_data/2023-03-18_2.2_raw_dataset_dict_nocut_only_one_text.txt
    # --data /data/zyx2022/FinanceText/process_file/2.1_raw_dataset_dict_nocut.txt \
    # --bert_save_dir /data2/panziyang/project/eccnlp/eccnlp/checkpoint/bert-chinese/$(date +%Y%m%d) \
    # --save_dir /data/pzy2022/project/eccnlp/checkpoint/$(date +%Y%m%d) \