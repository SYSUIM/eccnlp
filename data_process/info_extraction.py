import logging
import re
from utils import split_dataset


def data_filter(data_list: list) -> set:
    filted_data = set()
    for i in range(len(data_list)):
        if data_list[i]['label'] == 1:
            filted_data.add(data_list[i]['number'])
    
    return filted_data

def sentence_cut(filted_data: list) -> list:
    processed_sentence = []
    processed_data = []
    
    for i in range(len(filted_data)):
    # for i in range(10):
        if filted_data[i]['number'] in processed_sentence:
            continue

        processed_sentence.append(filted_data[i]['number'])

        split_rule = '[。？！]'
        # filter all empty string, and split the raw string based on "。？！"
        sub_str_list = filter(None, re.split(split_rule, filted_data[i]['raw_text'])[:-1])
        result_list = filted_data[i]['result_list']
        
        if result_list:
            for sub_str in sub_str_list:
                sample = {"content": sub_str, "raw_text": filted_data[i]['raw_text'], "number": filted_data[i]['number']}

                for result_dict in result_list:
                    result = result_dict['text']
                    if result not in sub_str:
                        flag = False
                    else:
                        flag = True
                
                if flag:
                    for result_dict in result_list:
                        result = result_dict['text']
                        if result in sub_str:
                            sample['result_list'] = [{"text": result, "start": sub_str.find(result), "end": sub_str.find(result) + len(result)}]
                            sample['prompt'] = "业绩归因"
                            processed_data.append(sample)
                else:
                    sample['result_list'] = [{"text": "", "start": 0, "end": 0}]
                    sample["prompt"] = ""
                    processed_data.append(sample)

        else:
            for sub_str in sub_str_list:
                sample = {
                    "content": sub_str,
                    "number": filted_data[i]['number'],
                    "raw_text": filted_data[i]['raw_text'],
                    "result_list": [{"text": "", "start": 0, "end": 0}],
                    "prompt": ""
                    }
                processed_data.append(sample)
    
    return processed_data


def data_process(filter_number: set, original_data: list):
    data_list = []
    # for number in filter_number:
    for data in original_data:
        if data['number'] in filter_number:
            # if data['label'] == 0:
            #     data['result_list'] = [{"text": "", "start": 0, "end": 0}]
            #     data['prompt'] = ''
            data_list.append(data)
    
    return data_list
                


def dataset_split(args, data_list: list):
    train_data = []
    dev_data = []
    test_data = []

    train_data, dev_data, test_data = split_dataset(data_list, args.train_size, args.val_size)

    return train_data, dev_data, test_data


def dataset_generate_train(args, data_list):
    # from utils import get_logger, get_log_path
    # ext_logger = get_logger('ext_logger', get_log_path() + '/ext.log')

    filted_data = data_filter(data_list)
    logging.info(f'raw_data length: {len(data_list)}, filted_data length: {len(filted_data)}')

    cutted_data = data_process(filted_data, data_list)
    logging.info(f'generate datasetlength: {len(cutted_data)}')

    train_data, dev_data, test_data = dataset_split(args, cutted_data)

    return train_data, dev_data, test_data


if __name__ == '__main__':
    train_data, dev_data, test_data = dataset_generate_train()
    # cutted_data =  dataset_generate()
    # print(len(train_data))
    # print(len(dev_data))
    # print(len(test_data))
    # print(len(cutted_data))
    # with open("./test.txt", 'w') as f:
    #     [f.write(str(data) + '\n') for data in cutted_data]