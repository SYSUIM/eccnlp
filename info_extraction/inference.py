import sys
sys.path.append('/data/pzy2022/project/eccnlp')


def extraction_inference(args, train_data, dev_data, test_data):
    from paddlenlp import Taskflow

    schema = ['业绩归因']
    my_ie = Taskflow("information_extraction", schema=schema, task_path = args.save_dir + '/model_best', position_prob = 0.9)

    for data in train_data:
        result = my_ie(data['content'])
        data['output'] = result

    for data in dev_data:
        result = my_ie(data['content'])
        data['output'] = result

    for data in test_data:
        result = my_ie(data['content'])
        data['output'] = result
    
    return train_data, dev_data, test_data