export CUDA_VISIBLE_DEVICES="0, 4"
nohup python -u ./train/train.py \
    --data /data/zyx2022/FinanceText/process_file/2.1_raw_dataset_dict_nocut.txt \
    --bert_pretrained_model /data/pzy2022/pretrained_model/bert-base-chinese \
    --bert_batch_size 16 \
    --bert_disable_tqdm False \
    --bert_save_dir /data2/panziyang/project/eccnlp/eccnlp/checkpoint/bert-chinese/$(date +%Y%m%d) \
    --balance up \
    --model EnsembleModel \
    --ensemble_models TextCNN TextRNN TextRCNN TextRNN_Att DPCNN \
    --num_epochs 10 \
    --batch_size 64 \
    --UIE_batch_size 4 \
    --UIE_num_epochs 2 \
    --device gpu:2 \
    --save_dir /data/pzy2022/project/eccnlp/checkpoint/20230228 
    # >/dev/null 2>&1 &
    # >>./log/$(date +%Y%m%d).log 2>&1 &