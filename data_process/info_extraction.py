import sys
import re

def read_file(path: str) -> list:
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            data_list.append(eval(line.strip('\n'), {'nan': ''}))

    return data_list

def data_filter(data_list: list) -> list:
    filted_data = []
    for i in range(len(data_list)):
        if data_list[i]['label'] == 1:
            filted_data.append(data_list[i])
    
    return filted_data

def sentence_cut(filted_data: list) -> list:
    processed_sentence = []
    processed_data = []
    
    for i in range(len(filted_data)):
    # for i in range(10):
        if filted_data[i]['raw_text'] in processed_sentence:
            continue

        processed_sentence.append(filted_data[i]['raw_text'])

        split_rule = '[。？！]'
        # filter all empty string, and split the raw string based on "。？！"
        sub_str_list = filter(None, re.split(split_rule, filted_data[i]['raw_text'])[:-1])
        result_list = filted_data[i]['result_list']
        
        if result_list:
            for sub_str in sub_str_list:
                sample = {"content": sub_str, "raw_text": filted_data[i]['raw_text']}

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
                    "raw_text": filted_data[i]['raw_text'],
                    "result_list": [{"text": "", "start": 0, "end": 0}],
                    "prompt": ""
                    }
                processed_data.append(sample)
    
    return processed_data

def data_process(data_list: list):
    train_data = []
    dev_data = []
    test_data = []
    
    for i in range(len(data_list)):
        if i % 5 == 1:
            dev_data.append(data_list[i])
        elif i % 5 == 2:
            test_data.append(data_list[i])
        else:
            train_data.append(data_list[i])

    return train_data, dev_data, test_data

def dataset_generate():
    path = '/data/pzy2022/project/eccnlp_local/info_extraction/result_data_TextRNN.txt'
    data_list = read_file(path)
    filted_data = data_filter(data_list)
    cutted_data = sentence_cut(filted_data)
    train_data, dev_data, test_data = data_process(cutted_data)

    return train_data, dev_data, test_data


if __name__ == '__main__':
    train_data, dev_data, test_data = dataset_generate()
    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))