export CUDA_VISIBLE_DEVICES="1,2,3,4"
nohup python -u ./train/train.py \
    --data /data/zyx2022/FinanceText/process_file/2.2_raw_dataset_dict_nocut_uni_no.txt \
    --bert_pretrained_model /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_batch_size 16 \
    --bert_disable_tqdm False \
    --bert_save_dir /data/fkj2023/Project/eccnlp/checkpoint/bert_chinese/$(date +%Y%m%d) \
    --balance up \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --num_epochs 20 \
    --batch_size 16 \
    --UIE_batch_size 8 \
    --UIE_num_epochs 50 \
    --rerank_epoch 200\
    --device gpu \
    --valid_steps 100 \
    --logging_steps 100 \
    --UIE_learning_rate 1e-6 \
    --save_dir /data/pzy2022/project/eccnlp/checkpoint/20230228 \
    --rerank_save_path /data/fkj2023/Project/eccnlp/checkpoint/rerank_model/04_01_15_38_12.pkl \
    >>./log/$(date +%Y%m%d)_2.2_uni_0228_train_rerank_compare_80.log 2>&1 &
    # >/dev/null 2>&1 &
    # >>./log/$(date +%Y%m%d).log 2>&1 &
    # >>./log/$(date +%Y%m%d)_2.1_nocut_train_test.log 2>&1 &
    #  /data/fkj2023/practice/eccnlp_data/2023-03-18_2.2_raw_dataset_dict_nocut_only_one_text.txt
    # --data /data/zyx2022/FinanceText/process_file/2.1_raw_dataset_dict_nocut.txt \
    # --bert_save_dir /data2/panziyang/project/eccnlp/eccnlp/checkpoint/bert-chinese/$(date +%Y%m%d) \
    # --save_dir /data/pzy2022/project/eccnlp/checkpoint/$(date +%Y%m%d) \
    # /data/fkj2023/practice/atest/cls_benchmark_kg_dict_all.txt
    # /data/zyx2022/FinanceText/process_file/2.2_raw_dataset_dict_nocut_uni_no.txt
    # --type 关键词 \
    # /data/pzy2022/project/eccnlp/checkpoint/20230228
    # --save_dir /data/fkj2023/Project/eccnlp/checkpoint/$(date +%Y%m%d) \
    # --save_dir /data/fkj2023/Project/eccnlp/checkpoint/20230324 \