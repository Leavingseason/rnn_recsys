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
from small_jobs import *
from helper.utils import try_get_param



def train_autoencoder():
    
    hparams = {
       'vocab_size': config.VOC_SIZE, 
       'emb_size': 128    ,
       'learning_rate': 0.001, #1e-3 for DAE
       'lambda_w': 0.0001,
       'batch_size': 256, 
       'is_tied_params':False,
       'alpha_enc': 0.1,
       'tf_summary_file': r'your_path\tf\events\autoencoder_' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') ,
       'sample_negative': True
    }
    
    model_type = 'CDAE' # DAE, CDAE
    batch_size = try_get_param(hparams, 'batch_size', 32)
    is_sample_negative = try_get_param(hparams, 'sample_negative', False)
    
    if model_type == 'CDAE':
        article_file = r'your_path\articles_CDAE.txt' 
        autoencoder = CDAE(**hparams)
        data_loader_func = data_loader.load_CDAE_data
        model_save_path = r'your_path\tf\models\CDAE.model'
    elif model_type == 'DAE':
        article_file =  r'your_path\articles_DAE.txt'
        autoencoder = SDAE(**hparams)
        data_loader_func = data_loader.load_autoencoder_data
        model_save_path = r'your_path\tf\models\DAE.model.'
    else:
        raise ValueError('unknow model type {0}'.format(model_type))
         
    #autoencoder.restore_model(r'your_path\tf\models\lr1e-3lambda1e-3\DAE.model. 9')  
    for epoch in range(utils.try_get_param(hparams, 'num_epoch',50)):
        
        epoch_start = clock() 
        
        epoch_loss = 0 
        epoch_error = 0 
        
        for batch_data in data_loader_func(article_file, batch_size = batch_size, sample_negative = is_sample_negative):
            cur_error, cur_loss =  *autoencoder.fit(*batch_data),
            epoch_loss += cur_loss 
            epoch_error += cur_error            
        
        epoch_end = clock() 
        
        config.logger.info('eopch: %d time: %.2f min training_error: %.2f loss: %.2f'  %(epoch, (epoch_end - epoch_start)/60.0, epoch_error, epoch_loss))
    
        autoencoder.save_model(model_save_path, epoch )


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
    
    # make sure the size of the last layer equals to the article embedding dimension
    hparams = {
       'dim': 5,
       'learning_rate':0.01  ,
       'lambda_w':0.00001,
       'layer_sizes': [5]  ,
       #'batch_size': 256, 
       'cell_type': 'rnn' , 
       'loss': 'log_loss',
       'num_epoches': 50 , 
       'eva_epoch' : 1 , 
       'tf_summary_file': r'tf\events\RS\RS_' + datetime.utcnow().strftime('%Y-%m-%d_%H_%M_%S') 
    }
    
    model_type = 'rnn'  # avg, rnn
    
    '''
    trainfile format: list of history\t target_article_id\t label
    article file format: article_id\t embeddings
    '''    
    
    article_emb_file = r'data/RS/articles_embeddings.txt'
    train_file = r'data/RS/train.txt'   # avg 0.6261
    valid_file = r'data/RS/train.txt'
    
    batch_size = utils.try_get_param(hparams, 'batch_size', 4)
    eva_epoch = utils.try_get_param(hparams, 'eva_epoch', 4)
    
    if model_type == 'avg':
        my_model = LinearAvgRS(**hparams)
        data_loader_func = data_loader.load_avgRS_data    
        model_save_path = r'tf\linearAVG\linearAvgRs.model'
    elif model_type == 'rnn':
        my_model = RNNRS(**hparams)
        data_loader_func = data_loader.load_rnnRS_data      
        model_save_path = r'tf\RNN\rnnRs.model'  
    else:
        raise ValueError("unsupported model name : {0}".format(model_type))

    eval_start = clock()
    metrics = evaluate_RS(my_model, data_loader_func, valid_file, article_emb_file, batch_size)
    eval_end = clock()
    config.logger.info('valid metrics at epoch {0}: {1}\ttime: {2:.2f} min'.format(-1, '  '.join(['{0}:{1:.4f}'.format(a,b) for a,b in metrics.items()]), (eval_end-eval_start)/60.0))
    

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
            

        my_model.save_model(model_save_path, epoch)
 
def encode_articles(outfile,
                    infile = r'Your_path\articles.txt', 
                    word_hashing_file = r'Your_path\articles_wordhashing_3w.obj', 
                    modelfile = r'Your_path\tf\models\CDAE\CDAE.model-14',
                    ):
 
    hparams = {
       'vocab_size': config.VOC_SIZE, 
       'emb_size': 128    ,
       'learning_rate': 0.001, #1e-3 for DAE
       'lambda_w': 0.001,
       'batch_size': 256, 
       'is_tied_params':False,
       'alpha_enc': 0.1, 
       'sample_negative': True
    }
    
    model_type = 'CDAE' # DAE, CDAE  
    
    if model_type == 'CDAE': 
        autoencoder = CDAE(**hparams)
        data_loader_func = data_loader.load_CDAE_data 
    elif model_type == 'DAE': 
        autoencoder = SDAE(**hparams)
        data_loader_func = data_loader.load_autoencoder_data 
    else:
        raise ValueError('unknow model type {0}'.format(model_type))  
    
    autoencoder.restore_model(modelfile)
     
    doc2title, doc2category = data_loader.load_documents(infile) 
    wh_model = utils.load_obj_from_file(word_hashing_file)
    
    
    doc_ids = list(doc2title.keys()) 
    with open(outfile, 'w') as wt:
        cnt = 0 
        for doc_id in doc_ids: 
            cur_title = doc2title[doc_id]
            cur_tfidf = utils.convert_line_to_tfidf(cur_title, wh_model, norm=True  )
            if not cur_tfidf:
                continue
 
            data_for_ae = data_loader.wrap_ae_data(cur_tfidf,0.2,True)
            encoding = autoencoder.get_encoding(  data_for_ae[0], data_for_ae[1], 1)   # consider using the original value instead of the noised values
            encoding =  np.reshape(encoding, [-1]).tolist() 
            wt.write('{0}\t{1}\n'.format(doc_id, ' '.join(['{0:.5}'.format(p) for p in encoding])))
            
            cnt+=1 
            if cnt%10000==0:
                print(cnt)
            
if __name__ == '__main__':    
    
	#Uncomment these lines and follow these steps to convert your raw data into our expected format. 
    
    #gen_word_hashing(r'YOUR_RAW_ARTICLE_FILE\articles.txt', r'articles_wordhashing_3w.obj', config.VOC_SIZE)
     
    r'''
    convert_raw_file_to_indexed(
        r'YOUR_RAW_ARTICLE_FILE\articles.txt', 
        r'path\articles_TFIDF_norm_3w.txt',  
        r'articles_wordhashing_3w.obj',
        norm=True      
    )
    '''
    
    #prepare_autoencoder_files(r'path\articles_TFIDF_norm_3w.txt', r'path\articles_DAE.txt')
    #prepare_triple_autoencoder_files(r'YOUR_RAW_ARTICLE_FILE\articles.txt', r'path\articles_TFIDF_norm_3w.txt', r'path\articles_CDAE.txt')
    
 
    #train_autoencoder() 
    
    #encode_articles(r'path\articles_embeddings.txt')
    
    train_RS()
        
