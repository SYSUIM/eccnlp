import sys
sys.path.append('/data/pzy2022/project/eccnlp')

# from data_process.info_extraction import read_file



def extraction_inference(args, test_data):
    from paddlenlp import Taskflow
    # path = '/data/pzy2022/project/eccnlp/data_process/after_classification_data3.1.txt'
    # data_list = read_file(path)

    schema = ['业绩归因']
    my_ie = Taskflow("information_extraction", schema=schema, task_path = args.save_dir + '/model_best', position_prob = 0.9)

    for i in range(len(test_data)):
    # for data in data_list:
        result = my_ie(test_data[i]['content'])
        test_data[i]['output'] = result
        # print(my_ie(data['content']))
    
    return test_data