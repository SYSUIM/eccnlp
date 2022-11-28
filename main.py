from data_process.dataprocess import re_pattern, train_dataset, predict_dataset, classification_dataset
from item_classification.run_classification_model import run_classification_model
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
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
args = parser.parse_args()

if __name__ == '__main__':
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