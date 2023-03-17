import random
import logging
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix

from utils import split_dataset, read_list_file


class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.to('cuda')
        self.labels = labels.to('cuda')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'input_ids': self.texts['input_ids'][idx],
                'attention_mask': self.texts['attention_mask'][idx],
                'labels': self.labels[idx]}


def my_loss_fn(outputs, labels):
    def loss_fn(outputs, labels):
        alpha, gamma = 0.25, 2
        BCE_loss = torch.nn.CrossEntropyLoss(reduction = 'none')(outputs, labels)
        pt = torch.exp(-BCE_loss)
        F_loss = alpha * (1-pt)** gamma * BCE_loss
        return torch.mean(F_loss)
    return loss_fn(outputs, labels)


class BertClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = my_loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def train_dataset_sampling(balance: str, train_X, train_y):
    print('Before resample labels ratio in dataset:', Counter(train_y))

    assert balance in ('up', 'down', 'none')

    if balance == 'none':
        train_X_resample, train_y_resample = train_X, train_y
        
    elif balance == 'up':
        sm = RandomOverSampler(random_state=42)
        train_X_resample, train_y_resample = sm.fit_resample(np.array(train_X).reshape(-1, 1), np.array(train_y).reshape(-1, 1))
        
    elif balance == 'down':
        sm = RandomUnderSampler(random_state=42)
        train_X_resample, train_y_resample = sm.fit_resample(np.array(train_X).reshape(-1, 1), np.array(train_y).reshape(-1, 1))
    
    return train_X_resample, train_y_resample


def generate_ids(tokenizer, texts: list, labels: list):
    text_ids = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512, 
        padding = 'max_length', 
        add_special_tokens=True
    )

    labels = torch.tensor(labels)
    
    return text_ids, labels


def test_on_best_model(tokenizer, test_dataset, test_model_path):
    # data preprocess for test_dataset
    test_texts, test_labels = [data['raw_text'] for data in test_dataset], [data['label'] for data in test_dataset]

    test_text_ids, test_labels = generate_ids(tokenizer, test_texts, test_labels)
    test_dataset = MyDataset(test_text_ids, test_labels)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 128,
        shuffle = False
    )
        
    test_model = BertForSequenceClassification.from_pretrained(
        test_model_path,
        torch_dtype = "auto", 
        num_labels = 2
    )
    print(f'using model: {test_model_path} on test dataset')

    predict_list, true_list = [], []
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            logits = test_model(**data).logits
            labels = data['labels'].to('cpu').numpy().tolist()
            predict = [(logit.argmax().item()) for logit in logits]

            true_list.extend(labels)
            predict_list.extend(predict)
    
    report = classification_report(true_list, predict_list)
    matrix = confusion_matrix(true_list, predict_list)
    
    return report, matrix


def BertForClassification(args, dataset):
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_pretrained_model)
    logging.info(f'tokenization start... using tokenizer from {args.bert_pretrained_model}')

    train_dataset, dev_dataset, test_dataset = split_dataset(dataset, args.train_size, args.val_size)

    # data preprocess for train_dataset
    train_X, train_y = [data['raw_text'] for data in train_dataset], [data['label'] for data in train_dataset]

    train_X_resample, train_y_resample = train_dataset_sampling(args.balance, train_X, train_y)

    train_X_resample = train_X_resample.reshape(-1).tolist()
    train_y_resample = train_y_resample.tolist()

    # random shuffle for resampled dataset
    sample_preshuffle = [(train_X_resample[i], train_y_resample[i]) for i in range(len(train_X_resample))]
    random.shuffle(sample_preshuffle)
    train_X_resample, train_y_resample = [text for text, _ in sample_preshuffle], [label for _, label in sample_preshuffle]
    print(f'your balance strategy: {args.balance}, after resample labels ratio in dataset: {Counter(train_y_resample)}')

    # data preprocess for dev_dataset
    dev_texts, dev_labels = [data['raw_text'] for data in dev_dataset], [data['label'] for data in dev_dataset]

    train_text_ids, train_labels = generate_ids(tokenizer, train_X_resample, train_y_resample)
    dev_text_ids, dev_labels = generate_ids(tokenizer, dev_texts, dev_labels)

    train_dataset = MyDataset(train_text_ids, train_labels)
    dev_dataset = MyDataset(dev_text_ids, dev_labels)

    model = BertForSequenceClassification.from_pretrained(
        args.bert_pretrained_model,
        torch_dtype = "auto", 
        num_labels = 2
        )
    print(f'using model from {args.bert_pretrained_model}')
    
    training_args = TrainingArguments(
        output_dir=args.bert_save_dir,
        load_best_model_at_end = True,
        save_strategy = "steps",
        evaluation_strategy = "steps",
        logging_steps = 100,
        learning_rate=2e-5,
        per_device_train_batch_size=args.bert_batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        weight_decay=0.01,
        disable_tqdm=True,
        save_steps = 100
    )

    trainer = BertClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    
    trainer.train()

    best_ckpt_path = trainer.state.best_model_checkpoint
    print(f'train completed, checkpoints saved in {best_ckpt_path}, the best model is {best_ckpt_path}')

    #evaluation on model_best
    report, matrix = test_on_best_model(tokenizer, test_dataset, best_ckpt_path)

    return report, matrix
    

if __name__ == '__main__':
    # the model_best may not be the exactly best model for the task, use code to test all models you ever saved
    raw_dataset = read_list_file('/data/zyx2022/FinanceText/process_file/2.1_raw_dataset_dict_nocut.txt')
    _, _, test_dataset = split_dataset(raw_dataset, 0.6, 0.2)

    tokenizer_path = '/data/pzy2022/pretrained_model/bert-base-chinese'
    test_model_path = '/data2/panziyang/project/eccnlp/eccnlp/checkpoint/bert-chinese/20230314/checkpoint-700'

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    logging.info(f'tokenization start... using tokenizer from {tokenizer_path}')

    report, matrix = test_on_best_model(tokenizer, test_dataset, test_model_path)
    print(report)
    print(matrix)