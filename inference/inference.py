import sys
sys.path.append('.')

import logging
from info_extraction.inference import extraction_inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def classification():
    return

def extraction(args, dataset):
    result = extraction_inference(args, dataset)
    return result

if __name__ == '__main__':
    args = config.get_arguments()

    dataset = read_list_file(args.data)
    logging.info(f'length of raw dataset: {len(dataset)}')

    # waiting for re filter...
    dataset = re_filter(dataset)
    logging.info(f'{len(dataset)} samples are filted by re_filter')
    
    extraction(args, dataset)