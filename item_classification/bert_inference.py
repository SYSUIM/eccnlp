from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
 
# print(torch.cuda.current_device(), torch.cuda.get_device_name())

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, ids, texts):
        self.texts = texts.to('cuda')
        self.ids = ids.to('cuda')
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {'input_ids': self.ids['input_ids'][idx],
                'attention_mask': self.ids['attention_mask'][idx],
                'raw_text': self.texts[idx]}


def generate_ids(args, data_list):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_path)

    ids = tokenizer(
        [data['raw_text'] for data in data_list],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len, 
        padding = 'max_length', 
        add_special_tokens=True
    )

    texts = torch.tensor([data['raw_text'] for data in data_list])
    
    return ids, texts


def bertForSequenceClassification(args, dataset):
    filtedDataset = []

    model = BertForSequenceClassification.from_pretrained(
        args.bert_model_path,
        torch_dtype = "auto"
        ).to('cuda')

    test_ids, test_texts = generate_ids(args, dataset)
    test_dataset = MyDataset(test_ids, test_texts)

    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = args.bert_batch_size,
        shuffle = False
        )
    
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            logits = model(**data).logits
            predicts = [(logit.argmax().item()) for logit in logits]

            filted_text = [data['raw_text'][i] for i in range(len(predicts)) if predicts[i] == 1]
            filtedDataset.extend(filted_text)

    return filtedDataset

