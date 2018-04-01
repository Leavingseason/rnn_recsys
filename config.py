'''
@author: v-lianji
'''

import logging
import tensorflow as tf 
from datetime import datetime

logging_filename =  datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') + '.log'
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(message)s')

logger.setLevel(logging.INFO)
#handler = logging.FileHandler(logging_filename)
#handler.setLevel(logging.INFO)
#handler.setFormatter(formatter)
#logger.addHandler(handler)


handler02 = logging.StreamHandler()
handler02.setLevel(logging.INFO)
handler02.setFormatter(formatter)
logger.addHandler(handler02)

logger.info('launching the program!')
logger.info(tf.__version__)