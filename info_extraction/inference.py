import logging

def extraction_inference(train_data, 
                         dev_data, 
                         test_data,
                         prompt,
                         save_dir,
                         position_prob):
    
    from paddlenlp import Taskflow

    # schema = ['业绩归因']
    schema = [prompt]
    my_ie = Taskflow("information_extraction", schema=schema, task_path = save_dir + '/model_best', position_prob = position_prob)

    logging.info(f'extraction start... using model from {save_dir}/model_best')
    if train_data is not None:    
        for data in train_data:
            result = my_ie(data['raw_text'])
            data['output'] = result

    if dev_data is not None:
        for data in dev_data:
            result = my_ie(data['raw_text'])
            data['output'] = result

    if test_data is not None:
        for data in test_data:
            result = my_ie(data['raw_text'])
            data['output'] = result
    logging.info(f'extraction end...')
    
    if train_data is not None: 
        logging.info(f'length of train_data:{len(train_data)}, dev_data: {len(dev_data)}, test_data: {len(test_data)}')
    else:
        logging.info(f'length of test_data: {len(test_data)}')
    
    return train_data, dev_data, test_data


if __name__ == '__main__':
    import sys
    sys.path.append('/data/pzy2022/project/eccnlp')
    import numpy as np

    from utils import read_list_file, accuracy_top_k, RR, AP
    from data_process.info_extraction import dataset_generate_train 

    # test_text = {'raw_text': '促进销售的根本在于回归市场和客户的原点，切实提高项目的市场竞争力，09年万科将从客户定位、产品设计、市场营销、工程质量、成本管理等方面入手，全面提升产品的竞争力。'}
    # test_data = [test_text]

    raw_dataset = read_list_file('/data/fkj2023/practice/eccnlp_data/2023-03-18_2.2_raw_dataset_dict_nocut_only_one_text.txt')
    train_data, dev_data, test_data = dataset_generate_train(0.6, 0.2, raw_dataset)

    _, _, test_result = extraction_inference(None, 
                         None, 
                         test_data,
                         prompt = '业绩归因',
                         save_dir = '/data/fkj2023/Project/eccnlp/checkpoint/20230325',
                        #  save_dir = '/data/pzy2022/project/eccnlp/checkpoint/20230228',
                         position_prob = 0.2)
    
    filted_result_on_test_data = [data for data in test_result if len(data['output'][0]) != 0]

    # accuracy_list = [accuracy_top_k(data, args.accuracy_k, args.type) for data in filted_result_on_test_data]
    accuracy_list_1 = [accuracy_top_k(data, k = 1, type = '业绩归因') for data in filted_result_on_test_data]
    accuracy_list_2 = [accuracy_top_k(data, k = 2, type = '业绩归因') for data in filted_result_on_test_data]
    accuracy_list_3 = [accuracy_top_k(data, k = 3, type = '业绩归因') for data in filted_result_on_test_data]
    accuracy_list_all = [accuracy_top_k(data, k = 20, type = '业绩归因') for data in filted_result_on_test_data]
    print(f'average accuracy@1 on filted_test_data_uie_res: {np.mean(accuracy_list_1)}')
    print(f'average accuracy@2 on filted_test_data_uie_res: {np.mean(accuracy_list_2)}')
    print(f'average accuracy@3 on filted_test_data_uie_res: {np.mean(accuracy_list_3)}')
    print(f'average accuracy@all on filted_test_data_uie_res: {np.mean(accuracy_list_all)}')
        
    rr = [RR(data, type = '业绩归因') for data in filted_result_on_test_data]
    print(f'MRR on filted_test_data_uie_res: {np.mean(rr)}')

    ap = [AP(data, type = '业绩归因') for data in filted_result_on_test_data]
    print(f'MAP on filted_test_data_uie_res: {np.mean(ap)}')
    
    # print(test_result)