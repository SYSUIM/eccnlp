import argparse
import config

# data preprocess
# from data_process.dataprocess import re_pattern, train_dataset, predict_dataset, classification_dataset

# item classification
from item_classification.run_classification_model import run_classification_model
from item_classification.ensemble_models import ensemble_classification_model
# from item_classification.ensemble_single_model import ensemble_single_model

# information extraction
from info_extraction.finetune import do_train
from data_process.info_extraction import dataset_generate

def text_classification(args):
    data = re_pattern(args)
    dataset = train_dataset(data, args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(dataset, args)
    predict_test = run_classification_model(train_list, dev_list, test_list, args)
    all_dict = []
    all_dict.extend(train_dict)
    all_dict.extend(val_dict)
    all_dict.extend(test_dict)
    # with open("./data/result_data/result_data_TextRNN.txt", 'w', encoding='utf8') as f:
    #     for i in range(len(all_dict)):
    #         f.write(str(all_dict[i]) + '\n')
    return all_dict


def ensemble_text_classification(args):
    data = re_pattern(args)
    dataset = train_dataset(data, args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(dataset, args)
    ensemble_classification_model(train_list, dev_list, test_list, args)
    # ensemble_single_model(train_list, dev_list, test_list, args)


def run_information_extraction(args, data):
    train_data, dev_data, test_data = dataset_generate()
    do_train(args, train_data, dev_data)
    return

if __name__ == '__main__':
    args = config.get_arguments()
    # all_dict = text_classification(args)
    # print(len(all_dict))
    # ensemble_text_classification(args)
    run_information_extraction(args, None)