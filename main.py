import argparse
import config

from data_process.dataprocess import re_pattern, train_dataset, predict_dataset, classification_dataset
from item_classification.run_classification_model import run_classification_model


def text_classification(args):
    data = re_pattern(args)
    dataset = train_dataset(data, args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(dataset, args)
    predict_test = run_classification_model(train_list, dev_list, test_list, args)
    all_dict = []
    all_dict.extend(train_dict)
    all_dict.extend(val_dict)
    all_dict.extend(test_dict)
    with open("./data/result_data/result_data.txt", 'w', encoding='utf8') as f:
        for i in range(len(all_dict)):
            f.write(str(all_dict[i]) + '\n')

if __name__ == '__main__':
    args = config.get_arguments()
    text_classification(args)