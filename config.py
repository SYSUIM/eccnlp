import argparse
import time

def get_arguments():
    # example for time_stamp: '2022_12_05_15_36'
    time_stamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    parser = argparse.ArgumentParser(description='eccnlp')

    '''
    here are parameters for Chinese Text Classification
    '''
    parser.add_argument('--data', type=str, required=True, help='path of raw data')
    parser.add_argument('--min_length', default=5, type=str, help='not less than 0')
    parser.add_argument('--balance', choices=["up", "down", "none"], default='none',type=str, required=True, help='up or down or none')
    parser.add_argument("--sampling_seed", default=10, type=int, help="Random seed for sampling data")
    # parser.add_argument('--predict_data', type=str, required=True, help='path of predict data')
    parser.add_argument("--classification_model_seed", default=1, type=int, help="Random seed for initializing classification model")
    parser.add_argument('--model', choices=["TextCNN", "TextRNN", "FastText", "TextRCNN", "TextRNN_Att", "DPCNN", "Transformer", "EnsembleModel"], type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
    parser.add_argument('--ensemble_models', type=str, nargs='+', help='ensemdble some models')
    # parser.add_argument('--num_ensemble', default=9, type=int, help='The number of ensembling single model')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='Random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--num_epochs', default=20, type=int, help='Total number of training epochs to perform.')
    parser.add_argument('--require_improvement', default=2000, type=int, help='Stop the train if the improvement is not required.')
    parser.add_argument('--n_vocab', default=0, type=int, help='Size of the vocab.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--pad_size', default=32, type=int, help='Size of the pad.')
    parser.add_argument('--clf_learning_rate', default=1e-3, type=float, help='The initial learning rate for Adam.')

    '''
    here are parameters for Information Extraction
    '''
    # for evaluate.py
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--model_path", type=str, default=None, help="The path of saved model that you want to load.")
    parser.add_argument("--test_path", type=str, default=None, help="The path of test set.")
    parser.add_argument("--UIE_batch_size", type=int, default=16, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--debug", action='store_true', help="Precision, recall and F1 score are calculated for each class separately if this option is enabled.")
    # --model_path ./checkpoint_short/model_6700 
    # --test_path ./data/test_data.txt 
    # --batch_size 16 
    # --max_seq_len 512 
    # --device gpu

    # for finetune.py
    # parser.add_argument("--UIE_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--UIE_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.") 
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    # parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length, Sequences longer than this will be split automatically.")
    parser.add_argument("--UIE_num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int, help="The interval steps to evaluate model performance.")
    # parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--UIE_model", choices=["uie-base", "uie-tiny", "uie-medium", "uie-mini", "uie-micro", "uie-nano"], default="uie-base", type=str, help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--model_dir", default="/data/pzy2022/paddlepaddle/taskflow/", type=str, help="The pretrained model dir for few-shot learning.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")
    # --train_path /data/pzy2022/project/test/test_data.txt 
    # --dev_path ./data/train_validation_test/dataset1/validation_data.txt 
    # --save_dir ./checkpoint/checkpoint3_20221017 
    # --learning_rate 1e-6 
    # --batch_size 8 
    # --max_seq_len 512 
    # --num_epochs 50 
    # --model uie-base 
    # --seed 1000 
    # --logging_steps 10
    # --valid_steps 100
    # --device gpu 

    '''
    here are parameters for Rerank
    '''
    parser.add_argument('--MODEL_PATH', type=str, default='./data_model_v8_lambdarank.ckpt',help='模型保存路径')
    parser.add_argument('--test_data_path', type=str, default='2022-12-03_test.log',help='测试集路径')
    parser.add_argument('--result_path', type=str, default='result1203.csv',help='结果文件路径')
    parser.add_argument('--reason_path', type=str, default='2022-12-03_reason_of_test.log',help='测试集原因路径')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    get_arguments()