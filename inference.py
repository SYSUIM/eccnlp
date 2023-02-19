import sys
sys.path.append('.')

import logging
from info_extraction.inference import extraction_inference

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(module)s[line:%(lineno)d]: >> %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def extraction():
    result = extraction_inference()
    return result

if __name__ == '__main__':
    extraction()