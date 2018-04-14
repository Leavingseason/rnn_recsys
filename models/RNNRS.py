'''
@author: v-lianji
'''
import tensorflow as tf 
import numpy as np 
import math
import config
from models.LinearAvgRS import BaseRS
from pip._vendor.webencodings.labels import LABELS
from sklearn import metrics
from sklearn.metrics import roc_auc_score

class RNNRS(BaseRS):
    '''
    classdocs
    '''

    def __init__(self,  **hparam ):
        '''
        layer_sizes :  number of hidden units in each layer.  len(layer_sizes) indicates the number of layers. Make sure the size of the last layer equals to the article embedding dimension.
        layer_activations: like ['tanh','relu'] 
        '''
        super().__init__(**hparam)
         
        self.layer_sizes =  hparam['layer_sizes'] if 'layer_sizes' in hparam else [64] 
        self.cell_type =  hparam['cell_type'] if 'cell_type' in hparam else 'rnn'
        self.layer_func = [self._get_activation_func(name) for name in hparam['layer_activations']] if 'layer_activations' in hparam else [self._get_activation_func('tanh')]*len(self.layer_sizes)
        self.dim = self.layer_sizes[-1]
        
        self.X_seq = tf.placeholder(tf.float32, shape=(None, None, self.dim)) # batch_size, time_steps, output_size
        self.Item = tf.placeholder(tf.float32, shape=(None, self.dim)) # batch_size, output_size
        self.Label = tf.placeholder(tf.float32, shape=(None,1))
        self.Len_seq = tf.placeholder(tf.int64, shape=(None))
        
        self.predictions, self.error, self.loss,  self.train_step, self.summary  = self._build_model()
        
        self._init_graph()
    
    def _get_a_cell(self, size, func):
        if self.cell_type == 'rnn':
            return tf.nn.rnn_cell.BasicRNNCell(num_units = size, activation = func)
        elif self.cell_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(num_units = size, activation = func) 
        elif self.cell_type == 'gru':
            return tf.nn.rnn_cell.GRUCell(num_units = size, activation = func) 
        else:
            raise ValueError('unknown rnn type. {0}'.format(self.cell_type))
        
    def _gather_last_output(self, data, seq_lens):
        this_range = tf.range(tf.cast(tf.shape(seq_lens)[0], dtype=tf.int64), dtype = tf.int64)
        indices = tf.stack([this_range, seq_lens-1], axis=1)
        return tf.gather_nd(data, indices)
    
    def _build_model(self):
        
        global_bias = tf.Variable(tf.truncated_normal([1], stddev=self.init_value*0.1, mean=0), dtype=tf.float32, name='glo_b')
        
        rnn_cell = tf.contrib.rnn.MultiRNNCell([self._get_a_cell(size, func) for (size, func) in zip(self.layer_sizes, self.layer_func)])
        
        output , _ = tf.nn.dynamic_rnn(rnn_cell, self.X_seq, sequence_length = self.Len_seq, dtype=tf.float32 )
        
        u_t = self._gather_last_output(output, self.Len_seq)
        u_t = tf.reshape(u_t, (-1, self.layer_sizes[-1]), name = 'user_embedding')
        
        preds = tf.sigmoid( tf.reduce_sum(tf.multiply(u_t, self.Item), 1, keepdims = True) + global_bias , name= 'prediction')  ##--
        
        #error = tf.reduce_mean(tf.losses.log_loss(predictions=preds, labels=self.Label), name='mean_log_loss')
        error = self._get_loss(preds, self.Label)
        loss = error
        
        train_step = self._optimize(error, tf.trainable_variables())  
        
        tf.summary.scalar('error', error)
        tf.summary.scalar('loss', loss)
        
        merged_summary = tf.summary.merge_all()
            
        return preds, error, loss, train_step, merged_summary

    def fit(self, user_history, target_items, labels, user_history_lens, max_len, batch_size):
        error, loss, summary, _ = self.sess.run(
                  [self.error, self.loss, self.summary, self.train_step], 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Label: labels,
                   self.Len_seq: user_history_lens
                   }
        )
        self.log_writer.add_summary(summary, self._glo_ite_counter)
        self._glo_ite_counter+=1
        return (error, loss)

    def pred(self, user_history, target_items, labels=None, user_history_lens = None, max_len = None, batch_size = None):    
        '''
        return a numpy list
        '''
        preds = self.sess.run(
                  self.predictions, 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Len_seq: user_history_lens
                   }
        )
        return preds.reshape(-1)
    
    def evaluate(self, user_history, target_items, labels, user_history_lens, max_len, batch_size):
        error, loss, preds  = self.sess.run(
                  [self.error, self.loss, self.predictions], 
                  {self.X_seq: user_history, 
                   self.Item: target_items,
                   self.Label: labels,
                   self.Len_seq: user_history_lens
                   }
        )
        return (error, loss, np.reshape(preds, (-1)), np.reshape(labels, (-1)))
        
  
        