import pandas as pd
import pickle
import sys
sys.path.append('/data/fkj2023/Project/eccnlp_1')
from utils import read_list_file

# file_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/3.1_2023-02-21_predict_M19_0220.txt'
file_path = '/data/fkj2023/Project/eccnlp_1/log/2023_04_28v1/3.1_uni_add_rerank.txt'
data_list = read_list_file(file_path)
df = pd.DataFrame()


for data in data_list:
    res_list = []
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
        # 'label': res_list,
        'rerank_all': data['rerank'],
        # 'score_all': data['score'],
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
df.to_excel('/data/fkj2023/Project/eccnlp_1/log/2023_04_28v1/result_3.1_2023_04_28.xlsx')

# nohup python generate_res_inference.py &