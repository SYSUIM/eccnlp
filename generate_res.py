import pandas as pd

from data_process.info_extraction import read_file

file_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2.0_2022-12-23_add_rerank.txt'
# file_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/test/add_rank.txt'
data_list = read_file(file_path)
df = pd.DataFrame()

for data in data_list:
    res_list = []
    for i in data['result_list']:
        if i[0]["text"] != '':
            res_list.append(i[0]["text"])
    res_str = ''.join(res_list)
    lenth = len(data['rerank'])
    rank = [0]*10
    for j in range(10):
        rank[j] = ''
        if(j<lenth):
            rank[j]=''.join(data['rerank'][j])
    temp_dict = {
        'number':data['number'],
        'raw_text': data['raw_text'],
        'label': res_str,
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

df.to_excel('result_2022_12_23.xlsx')