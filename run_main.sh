nohup python -u main.py \
    --data 0
    --model TextCNN > $(date +%Y%m%d).log 2>&1 &