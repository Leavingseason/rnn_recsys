'''
@author: v-lianji
'''

import config
from helper import *
import codecs  
from helper import utils
from collections import Counter
import math 
import random 
from operator import itemgetter
import numpy as np
from models.DAE import * 
from models.CDAE import *
from datetime import datetime
from helper.data_loader import *  


def convert_doc_tfidf(doc2title, wh_model):    
    doc2tfidf = {} 
    for p in doc2title.items():
        cur_tfidf = utils.convert_line_to_tfidf(p[1], wh_model)
        if cur_tfidf:
            doc2tfidf[p[0]] = p[1]    
    return doc2tfidf


def print_top_words():
    infile = r'Y:\BingNews\Zhongxia\articles.txt'
    word_hashing_file = r'Y:\BingNews\Zhongxia\my\articles_wordhashing_3w.obj'
    doc2title, doc2category = load_documents(infile) 
    wh_model = utils.load_obj_from_file(word_hashing_file)
    #doc2tfidf = convert_doc_tfidf(doc2title, wh_model)
    
    ae_model = load_autoencoder()
    
    doc_ids = list(doc2title.keys()) 
    while True:
        try :
            doc_id = random.choice(doc_ids)
            cur_title = doc2title[doc_id]
            cur_tfidf = utils.convert_line_to_tfidf(cur_title, wh_model, norm=True  )
            if not cur_tfidf:
                continue
            print('docid: {0}\t cate: {2} \t title: {1}'.format(doc_id, doc2title[doc_id], doc2category[doc_id]))
            sorted_tfidf = sorted(cur_tfidf, key = itemgetter(1), reverse = True)
            k = min(10, len(sorted_tfidf))
            str_gt = ['{0}:{1:.2f}'.format(wh_model.idx2word[p[0]],p[1]) for p in sorted_tfidf[0:k]]
            print('gt: ' +  ' '.join(str_gt))
            
            data_for_ae = wrap_ae_data(cur_tfidf,0.2,True)
            pred , masked_pred = ae_model.get_predictions(  *data_for_ae, 1)
            pred = list( enumerate(np.reshape(pred, [-1]).tolist()) )
            masked_pred = list( enumerate(np.reshape(masked_pred, [-1]).tolist()) )            
            masked_pred = [p for p in masked_pred if p[1]>0.001]
            
            pred = sorted(pred, key = itemgetter(1), reverse = True)
            masked_pred = sorted(masked_pred, key = itemgetter(1), reverse = True)
            k = min(10, len(masked_pred))
            str_pred_all = ['{0}:{1:.2f}'.format(wh_model.idx2word[p[0]],p[1]) for p in pred[0:k]]
            str_pred_masked = ['{0}:{1:.2f}'.format(wh_model.idx2word[p[0]],p[1]) for p in masked_pred[0:k]]
            print('pred_all: ' +  ' '.join(str_pred_all))
            print('pred_masked: ' +  ' '.join(str_pred_masked))
            
            var = input("press enter to continue... ")
        except KeyboardInterrupt:
            break 


def load_autoencoder():
        
    hparams = {
       'vocab_size': config.VOC_SIZE, 
       'emb_size': 128    ,
       'learning_rate': 0.001,
       'lambda_w': 0.001,
       'batch_size': 256, 
       'is_tied_params':False,
       'tf_summary_file': r'Y:\BingNews\Zhongxia\my\tf\events\autoencoder_' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') ,
       'sample_negative': True
    }
    
    
    model_type = 'CDAE' # DAE, CDAE

    if model_type == 'CDAE': 
        autoencoder = CDAE(**hparams) 
    elif model_type == 'DAE': 
        autoencoder = SDAE(**hparams) 

    autoencoder.restore_model(r'Y:\BingNews\Zhongxia\my\tf\models\CDAE\CDAE.model-14')
    
    return autoencoder

if __name__ == '__main__':
    
     
    print_top_words()
    
    r'''
    word_hashing_file = r'Y:\BingNews\Zhongxia\my\training_articles_wordhashing.obj' 
    wh_model = utils.load_obj_from_file(word_hashing_file)
    word2freq = [(p[0],p[1]) for p in wh_model.word2freq.items()]
    res = sorted(word2freq, key=itemgetter(1))
    print(res[70000:70010])
    '''
               