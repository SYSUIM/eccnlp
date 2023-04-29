export CUDA_VISIBLE_DEVICES="2,3, 4"
python -u ./inference/inference.py \
    --bert_tokenizer_path /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_model_path /data/fkj2023/Project/eccnlp_1/checkpoint/bert_chinese/20230428_test/checkpoint-200 \
    --bert_batch_size 512 \
    --data /data/fkj2023/Project/eccnlp_data/3.1_raw_dataset_dict_nocut.txt \
    --predict_data /data/xf2022/Projects/eccnlp_local/data/dataset/3.1_data_dict_length5_test.txt \
    --balance down \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --ensemble_date 03.07 \
    --num_epochs 10 \
    --save_dir /data/fkj2023/Project/eccnlp_1/checkpoint/20230427 \
    --device gpu \
    --rerank_save_path /data/fkj2023/Project/eccnlp_1/checkpoint/rerank_model/04_27_21_28_41.pkl \
    >>./log/$(date +%Y%m%d)_3.1_uni_inference.log 2>&1 &
    # >/dev/null 2>&1 &
    # --data /data/zyx2022/FinanceText/process_file/3.1_data_dict_length0.txt \ 
    # --data /data/zyx2022/FinanceText/process_file/3.1_raw_dataset_dict_nocut.txt \
    # /data/fkj2023/Project/eccnlp_local/data/3.1_raw_dataset_dict_nocut_test.txt
    # /data/zyx2022/FinanceText/process_file/2.2_raw_dataset_dict_nocut_uni_no.txt
    # --save_dir /data/pzy2022/project/eccnlp/checkpoint/20230228 \
    # --save_dir /data/fkj2023/Project/eccnlp/checkpoint/20230324 \
    # /data/fkj2023/Project/eccnlp_1/checkpoint/bert_chinese/20230427/checkpoint-300
    # --rerank_save_path /data/fkj2023/Project/eccnlp_1/checkpoint/rerank_model/04_04_20_43_09.pkl \
    # --data /data/fkj2023/Project/eccnlp_data/2023-04-26_2.3_raw_dataset_dict_nocut_one_more_text.txt \
    # --data /data/fkj2023/Project/eccnlp_data/3.1_raw_dataset_dict_nocut.txt \
    # --rerank_save_path /data/fkj2023/Project/eccnlp_1/checkpoint/rerank_model/04_27_07_27_45.pkl \
    # checkpoint/rerank_model/04_27_21_28_41.pkl
    # /data/fkj2023/Project/eccnlp_1/checkpoint/bert_chinese/20230428_test/checkpoint-200
    # --bert_model_path /data/fkj2023/Project/eccnlp_1/checkpoint/bert_chinese/20230427/checkpoint-500 \
    # --bert_model_path /data2/panziyang/project/bertForSequenceClassification/checkpoint/bert-base-chinese_focalloss_nofilter_over_raw/checkpoint-14000 \