import pandas as pd

from info_extraction.preprocess import read_file

file_path = '/data/pzy2022/project/eccnlp_local/2022-12-15_add_rerank.txt'

data_list = read_file(file_path)
df = pd.DataFrame()
# print(data_list[0]['content'])

for data in data_list:
    temp_dict = {
        'content': data['content'],
        'label': data['result_list'],
        'output': data['output'],
        'rerank': data['rerank']
    }
    df = df.append(temp_dict, ignore_index=True)

df.to_excel('result_20221217.xlsx')