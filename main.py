import argparse

from data_process.dataprocess import re_pattern, train_dataset, predict_dataset, classification_dataset
from item_classification.run_classification_model import run_classification_model


parser = argparse.ArgumentParser(description='Chinese Text Classification')

'''
here are parameters for Chinese Text Classification
'''
parser.add_argument('--data', type=str, required=True, help='path of raw data')
parser.add_argument('--min_length', default=5, type=str, help='not less than 0')
parser.add_argument('--type', default='业绩归因', type=str, help='type of answer')
parser.add_argument('--balance', default='none',type=str, required=True, help='up or down or none')
# parser.add_argument('--predict_data', type=str, required=True, help='path of predict data')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--num_epochs', default=20, type=int, help='Total number of training epochs to perform.')
parser.add_argument('--require_improvement', default=2000, type=int, help='Stop the train if the improvement is not required.')
parser.add_argument('--n_vocab', default=0, type=int, help='Size of the vocab.')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU/CPU for training.')
parser.add_argument('--pad_size', default=32, type=int, help='Size of the pad.')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='The initial learning rate for Adam.')
parser.add_argument('--MODEL_PATH', type=str, default='./data_model_v8_lambdarank.ckpt',help='模型保存路径')
parser.add_argument('--test_data_path', type=str, default='2022-12-03_test.log',help='测试集路径')
parser.add_argument('--result_path', type=str, default='result1203.csv',help='结果文件路径')
parser.add_argument('--reason_path', type=str, default='2022-12-03_reason_of_test.log',help='测试集原因路径')

'''
here are parameters for Information Extraction
'''

args = parser.parse_args()

def text_classification(args):
    data = re_pattern(args)
    dataset = train_dataset(data, args)
    train_list, dev_list, test_list, train_dict, val_dict, test_dict = classification_dataset(dataset, args)
    predict_test = run_classification_model(train_list, dev_list, test_list, args)
    all_dict = []
    all_dict.extend(train_dict)
    all_dict.extend(val_dict)
    all_dict.extend(test_dict)
    with open("./data/result_data/result_data.txt", 'w', encoding='utf8') as f:
        for i in range(len(all_dict)):
            f.write(str(all_dict[i]) + '\n')

if __name__ == '__main__':
    text_classification(args)