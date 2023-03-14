import pandas as pd
import pickle

from data_process.info_extraction import read_file

file_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2022-12-23_add_rerank.txt'
file_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2023-01-15_predict-M15-0113_test.txt'
raw_path = '/data/pzy2022/project/eccnlp_local/2.0_raw_dict.pkl'
# file_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/add_rank.txt'
data_list = read_file(file_path)
df = pd.DataFrame()

with open(raw_path, 'rb') as f:
    raw_data = pickle.load(f)
# print(raw_data[572926])
# exit(0)

for data in data_list:
    res_list = []
    for reason in raw_data[data['number']]['result_list']:
        res_list.append(reason['text'])
    # for i in data['result_list']:
    #     if i[0]["text"] != '':
    #         res_list.append(i[0]["text"])
    # res_str = ''.join(res_list)
    lenth = len(data['rerank'])
    rank = [0]*10
    for j in range(10):
        rank[j] = ''
        if(j<lenth):
            rank[j]=''.join(data['rerank'][j])
    temp_dict = {
        'number':data['number'],
        'raw_text': data['raw_text'],
        # 'label': res_str,
        'label': res_list,
        'rerank_all': data['rerank'],
        'rank1': rank[0],
        'rank2': rank[1],
        'rank3': rank[2],
        'rank4': rank[3],
        'rank5': rank[4],
        'rank6': rank[5],
        'rank7': rank[6],
        'rank8': rank[7],
        'rank9': rank[8],
        'rank10': rank[9]
    }
    df = df.append(temp_dict, ignore_index=True)

# print(df)
df.to_excel('result_2023_01_191.xlsx')