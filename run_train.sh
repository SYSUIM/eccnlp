export CUDA_VISIBLE_DEVICES="1,2,3"
python -u ./train/train.py \
    --data /data/fkj2023/Project/eccnlp_data/2023-04-26_2.3_raw_dataset_dict_nocut_one_more_text.txt \
    --bert_pretrained_model /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_batch_size 16 \
    --bert_disable_tqdm False \
    --bert_save_dir /data/fkj2023/Project/eccnlp_1/checkpoint/bert_chinese/$(date +%Y%m%d_%H_%M)_test \
    --balance up \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --batch_size 16 \
    --UIE_batch_size 4 \
    --UIE_num_epochs 50 \
    --rerank_epoch 400\
    --device gpu \
    --valid_steps 100 \
    --logging_steps 100 \
    --UIE_learning_rate 1e-6 \
    --bert_epochs 10 \
    --save_dir /data/fkj2023/Project/eccnlp_1/checkpoint/$(date +%Y%m%d_%H_%M)_test \
    --rerank_learning_rate 1e-4 \
    >>./log/$(date +%Y%m%d)_2.3_uni_data_train_model_test.log 2>&1 &
    # >>./log/$(date +%Y%m%d)_2.3_test_eva_model.log 2>&1 &
    # >>./log/$(date +%Y%m%d)_2.3_uni_data_train_model_test_bert.log 2>&1 &
    # >>./log/$(date +%Y%m%d)_2.2_uni_0228_train_rerank_test_nobert.log 2>&1 &
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
    # --rerank_save_path /data/fkj2023/Project/eccnlp/checkpoint/rerank_model/04_01_15_38_12.pkl \
    # --rerank_save_path /data/fkj2023/Project/eccnlp/checkpoint/rerank_model/04_04_20_43_09.pkl \   #只使用context feature best
    # --rerank_save_path /data/fkj2023/Project/eccnlp_1/checkpoint/rerank_model/04_06_17_33_20.pkl    # 全部特征训练best
    # --save_dir /data/pzy2022/project/eccnlp/checkpoint/20230228 \
    # --rerank_save_path /data/fkj2023/Project/eccnlp_1/checkpoint/rerank_model/04_06_17_33_20.pkl \
    # --rerank_save_path /data/fkj2023/Project/eccnlp_1/checkpoint/rerank_model/04_27_07_27_45.pkl \