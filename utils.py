import os
import logging

logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
     datefmt='%Y-%m-%d %H:%M:%S'
     )

def read_list_file(path: str) -> list:
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            data_list.append(eval(line.strip('\n'), {'nan': ''}))
    logging.info(f'read data_list DONE. Length: {len(data_list)}')

    return data_list

if __name__ == '__main__':
    path = '/data/pzy2022/project/eccnlp/data_process/after_classification_data3.1.txt'
    read_list_file(path)