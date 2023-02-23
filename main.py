import os
import argparse
import logging
from multiprocessing import Process

from utils import read_list_file, evaluate_sentence
import config
from config import re_filter

# data preprocess
from data_process.dataprocess import re_pattern1, re_pattern2, train_dataset, split_dataset, classification_dataset

# item classification
from item_classification.run_classification_model import run_classification_model
from item_classification.ensemble_models import ensemble_classification_model
# from item_classification.ensemble_single_model import ensemble_single_model

# information extraction
from info_extraction.finetune import do_train
from info_extraction.inference import extraction_inference
from data_process.info_extraction import dataset_generate_train


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def text_classification(args):
    data = re_pattern1(args)
    dataset = train_dataset(data, args)

    # p_data = re_pattern11(args)
    # p_dataset = predict_dataset(p_data, args)
    # predict_list = dict_to_list(p_dataset)

    train_dict,val_dict,test_dict = split_dataset(dataset,args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(train_dict,val_dict,test_dict, args)
    # predict_test = run_classification_model(train_list, dev_list, predict_list, args)
    # for i in range(len(predict_test)):
    #     test_dict[i]['label'] = predict_test[i]
    # all_dict = []
    # all_dict.extend(train_dict)
    # all_dict.extend(val_dict)
    # all_dict.extend(test_dict)
    # return all_dict
    # for i in range(len(predict_test)):
    #     p_dataset[i]['label'] = predict_test[i]
    # # with open("./data/result_data/result_data_TextRNN.txt", 'w', encoding='utf8') as f:
    # #     for i in range(len(all_dict)):
    # #         f.write(str(all_dict[i]) + '\n')
    # return p_dataset


def ensemble_text_classification(args):
    data = re_pattern1(args)
    dataset = train_dataset(data, args)
    train_dict, val_dict, test_dict = split_dataset(dataset,args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(train_dict,val_dict,test_dict, args)
    predict_test = ensemble_classification_model(train_list, dev_list, test_list, args)
    # ensemble_single_model(train_list, dev_list, test_list, args)
    for i in range(len(predict_test)):
        test_dict[i]['label'] = predict_test[i]
    all_dict = []
    # all_dict.extend(train_dict)
    # all_dict.extend(val_dict)
    all_dict.extend(test_dict)
    # with open("./data/result_data/result_data_TextRNN.txt", 'w', encoding='utf8') as f:
    #     for i in range(len(all_dict)):
    #         f.write(str(all_dict[i]) + '\n')
    return all_dict


def run_information_extraction(args, data):
    train_data, dev_data, test_data = dataset_generate_train(args, data)
    logging.info(f'train_data: {len(train_data)}, dev_data: {len(dev_data)}, test_data: {len(test_data)}')
    do_train(args, train_data, dev_data)
    result_on_test_data = extraction_inference(args, test_data)
    
    return result_on_test_data

if __name__ == '__main__':
    args = config.get_arguments()

    dataset = read_list_file(args.data)

    # waiting for re filter...
    dataset = re_filter(dataset)
    
    # all_dict = text_classification(args)
    all_dict = ensemble_text_classification(args)
    logging.info('ensemble_text_classification training completed.')

    run_information_extraction(args, dataset)

    print('parent pid: ', os.getpid())
    processes = [
        Process(target = ensemble_text_classification, args = (args)),
        Process(target = run_information_extraction, args = (args, dataset))
    ]
    [p.start() for p in processes]
    [p.join() for p in processes]
    classification_result, extraction_result = [p.get() for p in processes]

    # evaluate for sentences after extraction
    evaluate_sentence(extraction_result, classification_result)