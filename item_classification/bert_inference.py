from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, BertTokenizerFast
import torch
 
# print(torch.cuda.current_device(), torch.cuda.get_device_name())

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, ids, numbers):
        self.numbers = numbers.to('cuda')
        self.ids = ids.to('cuda')
        
    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self, idx):
        return {'input_ids': self.ids['input_ids'][idx],
                'attention_mask': self.ids['attention_mask'][idx],
                'numbers': self.numbers[idx]}


def generate_ids(args, data_list):
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_tokenizer_path)

    ids = tokenizer(
        [data['raw_text'] for data in data_list],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len, 
        padding = 'max_length', 
        add_special_tokens=True
    )

    numbers = torch.tensor([data['number'] for data in data_list])
    
    return ids, numbers


def bertForSequenceClassification(args, dataset):
    filtedDataset = []
    number2dict = {}
    for data in dataset:
        number2dict[data['number']] = data

    model = BertForSequenceClassification.from_pretrained(
        args.bert_model_path,
        torch_dtype = "auto"
        ).to('cuda')

    test_ids, test_numbers = generate_ids(args, dataset)
    test_dataset = MyDataset(test_ids, test_numbers)

    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = args.bert_batch_size,
        shuffle = False
        )
    
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            input_keys = ('input_ids', 'attention_mask')
            input = {key: value for key, value in data.items() if key in input_keys}

            logits = model(**input).logits
            predicts = [(logit.argmax().item()) for logit in logits]

            numbers = data['numbers'].to('cpu').numpy().tolist()

            filted_text = [number2dict[numbers[i]] for i in range(len(predicts)) if predicts[i] == 1]
            filtedDataset.extend(filted_text)

    return filtedDataset    