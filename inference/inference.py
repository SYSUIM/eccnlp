import sys
sys.path.append('/data/pzy2022/project/eccnlp')
import config
from config import re_filter

from utils import read_list_file, split_dataset, evaluate_sentence

import logging
from info_extraction.inference import extraction_inference

from data_process.info_extraction import dataset_generate_train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def classification():
    return

def extraction(args, dataset):
    result = extraction_inference(args, dataset)
    return result

if __name__ == '__main__':
    args = config.get_arguments()

    dataset = read_list_file(args.data)
    # clf = read_list_file('/data/xf2022/Projects/eccnlp_local/data/result_data/result_data2.0_TextRCNN2022_12_21_15_37.txt')

    logging.info(f'length of raw dataset: {len(dataset)}')

    dataset = re_filter(dataset)
    train_data, dev_data, test_data = dataset_generate_train(args, dataset)
    logging.info(f'{len(dataset)} samples left after re_filter')
    
    result = extraction(args, test_data)
    # evaluate_sentence(result, clf)