'''
@author: v-lianji
'''

import config
import tensorflow as tf 
from models.DAE import * 
from models.CDAE import *
from models.LinearAvgRS import *
from models.RNNRS import *
import numpy as np 
from time import clock
from datetime import datetime
from helper import *




def train_autoencoder():
    
    hparams = {
       'vocab_size': 100000, 
       'emb_size': 10    ,
       'tf_summary_file': r'D:\My Projects\data\news\RS\logs\RS_' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') 
    }
    
    model_type = 'CDAE' # DAE
    
    if model_type == 'CDAE':
        article_file = r'D:\My Projects\data\news\articles_CDAE.txt' 
        autoencoder = CDAE(**hparams)
        data_loader_func = data_loader.load_CDAE_data
        model_save_path = r'D:\My Projects\data\news\processed\CDAE.model'
    elif model_type == 'DAE':
        article_file = r'D:\My Projects\data\news\articles.txt' 
        autoencoder = SDAE(**hparams)
        data_loader_func = data_loader.load_autoencoder_data
        model_save_path = r'D:\My Projects\data\news\processed\DAE.model'
    else:
        raise ValueError('unknow model type {0}'.format(model_type))
        
    #article_file = r'D:\My Projects\data\news\processed\artitle_title_2017-01-01-2017-01-31_word_index.txt'
      
    for epoch in range(utils.try_get_param(hparams, 'num_epoch',10)):
        
        epoch_start = clock() 
        
        epoch_loss = 0 
        epoch_error = 0 
        
        for batch_data in data_loader_func(article_file, batch_size = 4):
            cur_error, cur_loss =  *autoencoder.fit(*batch_data),
            epoch_loss += cur_loss 
            epoch_error += cur_error            
        
        epoch_end = clock() 
        
        config.logger.info('eopch: %d time: %.2f min training_error: %.2f loss: %.2f'  %(epoch, (epoch_end - epoch_start)/60.0, epoch_error, epoch_loss))
    
    autoencoder.save_model(model_save_path)


def evaluate_RS(my_model, data_loader_func, train_file, article_emb_file, batch_size):
    res ={} 
    
    #AUC:
    preds = []
    gts = []
    for batch_data in data_loader_func(train_file, article_emb_file, batch_size):
        batch_preds = my_model.evaluate(*batch_data)
        preds.append(batch_preds[2])
        gts.append(batch_preds[3])
    preds = np.concatenate(preds,axis=0)
    gts = np.concatenate(gts,axis=0)
    auc = roc_auc_score(np.asarray(gts, dtype=np.int64),preds)    
    res['auc'] = auc    
    # end of AUC
    
    return res

def train_RS():
    
    hparams = {
       'dim': 5,
       'learning_rate':0.01  ,
       'layer_sizes': [5]  ,
       'batch_size': 4, 
       'cell_type': 'rnn' , 
       
       'num_epoches': 50 , 
       'eva_epoch' : 2 , 
       'tf_summary_file': r'D:\My Projects\data\news\RS\logs\RS_' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') 
    }
    
    model_type = 'rnn'  # avg, rnn
    
    '''
    trainfile format: list of history\t target_article_id\t label
    article file format: article_id\t embeddings
    '''    
    
    article_emb_file = r'D:\My Projects\data\news\RS\article_embeddings.txt'  
    train_file = r'D:\My Projects\data\news\RS\train.txt'  
    valid_file = train_file
    
    batch_size = utils.try_get_param(hparams, 'batch_size', 4)
    eva_epoch = utils.try_get_param(hparams, 'eva_epoch', 4)
    
    if model_type == 'avg':
        my_model = LinearAvgRS(**hparams)
        data_loader_func = data_loader.load_avgRS_data    
        model_save_path = r'D:\My Projects\data\news\RS\linearAvgRs.model'
    elif model_type == 'rnn':
        my_model = RNNRS(**hparams)
        data_loader_func = data_loader.load_rnnRS_data      
        model_save_path = r'D:\My Projects\data\news\RS\rnnRs.model'  
    else:
        raise ValueError("unsupported model name : {0}".format(model_type))

    for epoch in range(utils.try_get_param(hparams, 'num_epoches', 10)):        
        epoch_start = clock()         
        epoch_loss = 0 
        epoch_error = 0 
        
        for batch_data in data_loader_func(train_file, article_emb_file, batch_size):
            cur_error, cur_loss = *my_model.fit(*batch_data),
            epoch_loss += cur_loss 
            epoch_error += cur_error            
        
        epoch_end = clock()         
        config.logger.info('eopch: %d time: %.2f min training_error: %.2f loss: %.2f'  %(epoch, (epoch_end - epoch_start)/60.0, epoch_error, epoch_loss))
        
        if epoch % eva_epoch == 0:
            eval_start = clock()
            metrics = evaluate_RS(my_model, data_loader_func, valid_file, article_emb_file, batch_size)
            eval_end = clock()
            config.logger.info('valid metrics at epoch {0}: {1}\ttime: {2:.2f} min'.format(epoch, '  '.join(['{0}:{1:.4f}'.format(a,b) for a,b in metrics.items()]), (eval_end-eval_start)/60.0))
            

    my_model.save_model(model_save_path)
            
if __name__ == '__main__':
    train_autoencoder()
    #train_RS()
        
