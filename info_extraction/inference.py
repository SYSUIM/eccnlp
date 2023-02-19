import sys
sys.path.append('/data/pzy2022/project/eccnlp')

from data_process.info_extraction import read_file
from paddlenlp import Taskflow


def extraction_inference():
    path = '/data/pzy2022/project/eccnlp/data_process/after_classification_data3.1.txt'
    data_list = read_file(path)

    schema = ['业绩归因']
    my_ie = Taskflow("information_extraction", schema=schema, task_path='/data/pzy2022/project/eccnlp/checkpoint/20230113/model_best', position_prob = 0.2)

    for i in range(len(data_list)):
    # for data in data_list:
        result = my_ie(data_list[i]['content'])
        data_list[i]['output'] = result
        # print(my_ie(data['content']))

    # with open("./after_extraction_data3.1.txt", 'w') as f:
    #     [f.write(str(data) + '\n') for data in data_list]