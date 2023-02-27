import pandas as pd
from sklearn.metrics import classification_report

from data_process.info_extraction import read_file

def read(true_path: str, predict_path: str):
    return read_file(true_path), read_file(predict_path)

def compare(true_list: list, predict_list: list):
    predict_index_list, true_index_list = {}, []
    for predict in predict_list:
        predict_index_list[predict['number']] = predict['rerank']

    for true in true_list:
        true_index_list.append(true['number'])
    # print(len(true_index_list))
    # print(len(set(true_index_list)))

    # ensure unduplicated list be created, otherwise exit
    if len(predict_index_list) != len(set(predict_index_list)):
        print('len(predict_index_list):', len(predict_index_list))
        print('len(set(predict_index_list)):', len(set(predict_index_list)))
        exit(0)

    result_df = pd.DataFrame()
    # # print(predict_index_list)
    # for true in true_list:
    #     # print(true['number'])
    #     if true['number'] == 530806:
    #         print(type(true['number']))
    #     # exit(0)
    # exit(0)

    # print(type(set(predict_index_list.keys())))
    # print(len(set(predict_index_list.keys())))
    # print(len(set(predict_index_list.keys()) & set(true_index_list)))
    # exit(0)

    flag = 0
    proceseed_num = []
    list1, list2 = [], []
    for true in true_list:
        index = true['number']
        if index in proceseed_num:
            continue
        if index in predict_index_list.keys():
            true_flag, predict_flag = None, None
            proceseed_num.append(index)
            if not true['prompt']:
                true_flag = 'negetive'
                if not predict_index_list[index][0]:
                    predict_flag = 'negetive'
                else:
                    predict_flag = 'positive'
                # flag += 1
                # print(true['result_list'], 'negetive')
            else:
                true_flag = 'positive'
                for reason in true['result_list']:
                    if (str(predict_index_list[index][0]) in reason['text']) or (reason['text'] in str(predict_index_list[index][0])):
                        predict_flag = 'positive'
                        break
                if predict_flag is None:
                    predict_flag = 'negetive'
            print(index, true_flag, predict_flag)
            print(true['result_list'], predict_index_list[index][0])
            list1.append(true_flag)
            list2.append(predict_flag)
            # if not predict_index_list[index][0]:
            #     flag += 1
            #     predict_flag = 'negetive'
            #     # print(predict_index_list[index], 'negetive')
            #     if 
            # else:
            #     predict_flag = 'negetive'
    # print(flag)
    print('--------------- account-type ---------------')
    classification_report_actual = classification_report(list1, list2)
    print(classification_report_actual)
    exit(0)

    for true in true_list:
        if true['number'] in predict_index_list.keys():
            flag += 1
    print(flag)
    exit(0)

    for index in predict_index_list.keys():
        true_flag, predict_flag = None, None
        if not predict_index_list[index][0]:
            predict_flag = 'negative'

        for true in true_list:
            if index == true['number']:
                if not true['prompt']:
                    true_flag = 'negative'
                else:
                    true_flag = 'positive'
                    if not predict_index_list[index][0]:
                        break
                    for reason in true['result_list']:
                        if (str(predict_index_list[index][0]) in reason['text']) or (reason['text'] in str(predict_index_list[index][0])):
                            predict_flag = 'positive'
                            break
                    predict_flag = 'negative'
                print([index, true_flag, predict_flag])
                flag += 1
                result_df = result_df.append([[index, true_flag, predict_flag]])
                break
    
    result_df.columns = ['index_number', 'true_flag', 'predict_flag']
    print(result_df)
    print(flag)

def main():
    true_path = '/data/pzy2022/project/eccnlp_local/2.0_raw_dict.txt'
    predict_path = '/data/fkj2023/Project/eccnlp_local/phrase_rerank/data/res_log/2023-01-15_predict-M15-0113_test.txt'
    true_list, predict_list = read(true_path, predict_path)
    compare(true_list, predict_list)

if __name__ == '__main__':
    main()