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
    
    test_text = {'raw_text': '促进销售的根本在于回归市场和客户的原点，切实提高项目的市场竞争力，09年万科将从客户定位、产品设计、市场营销、工程质量、成本管理等方面入手，全面提升产品的竞争力。'}
    test_data = [test_text]

    _, _, test_result = extraction_inference(None, 
                         None, 
                         test_data,
                         prompt = '业绩归因',
                        #  save_dir = '/data/fkj2023/Project/eccnlp/checkpoint/20230322',
                         save_dir = '/data/pzy2022/project/eccnlp/checkpoint/20230228',
                         position_prob = 0.2)
    
    print(test_result)