'''
@author: v-lianji
'''

import tensorflow as tf 
import numpy as np 
import math
import config

'''
sparse denoising autoencoder
sparse input, all input values should be >=0. input value = 0 indicates a missing value 
'''
class SDAE(object):
    
    def __init__(self, **hparam ):
        '''
        vocab_size, emb_size, enc_func, dec_func, is_tied_params, lambda_w, learning_rate, type_of_opt
        (adadelta)rho, (adam)beta1, beta2, epsilon
        '''
        self.vocab_size = hparam['vocab_size'] if 'vocab_size' in hparam else 100000 
        self.emb_size = hparam['emb_size'] if 'emb_size' in hparam else 64
        self.is_tied_params = hparam['is_tied_params'] if 'is_tied_params' in hparam else False
        self.init_value = hparam['init_value'] if 'init_value' in hparam else 0.01
        self.lambda_w = hparam['lambda_w'] if 'lambda_w' in hparam else 0.001 
        self.lr = hparam['learning_rate'] if 'learning_rate' in hparam else 0.001 
        self.opt = hparam['type_of_opt'] if 'type_of_opt' in hparam else 'adam'
        self.rho = hparam['rho'] if 'rho' in hparam else 0.95
        self.epsilon = hparam['epsilon'] if 'epsilon' in hparam else 1e-8
        self.beta1 = hparam['beta1'] if 'beta1' in hparam else 0.9
        self.beta2 = hparam['beta2'] if 'beta2' in hparam else 0.999
        
        self.enc_func =  self.get_activation_func(hparam['enc_func'] if 'enc_func' in hparam else 'tanh')
        self.dec_func =  self.get_activation_func(hparam['dec_func'] if 'dec_func' in hparam else 'tanh')
        
        self.summary_path = hparam['tf_summary_file'] if 'tf_summary_file' in hparam else 'log_tmp_path'

        self.saver = None 
        
        self.X = tf.sparse_placeholder(tf.float32) 
        self.Y = tf.sparse_placeholder(tf.float32) 
        self.mask = tf.sparse_placeholder(tf.float32) 
        
        self.params = {}
        
        self.W = tf.Variable(
                            tf.truncated_normal([self.vocab_size, self.emb_size], stddev=self.init_value / math.sqrt(float(self.emb_size)), mean=0), 
                            name='encoder_W' , dtype=tf.float32
                            )
        self.b = tf.Variable(tf.truncated_normal([self.emb_size], stddev=self.init_value * 0.001, mean=0), name='encoder_bias', dtype=tf.float32 )
        
        self.params['W'] = self.W 
        self.params['b'] = self.b 
            
        if not self.is_tied_params:
            self.W_prime = tf.Variable(
                            tf.truncated_normal([self.emb_size, self.vocab_size], stddev=self.init_value / math.sqrt(float(self.emb_size)), mean=0),  
                            name='decoder_W' , dtype=tf.float32
                            ) 
            self.params['W_prime'] = self.W_prime 
        else:
            self.W_prime = tf.transpose(self.W) 
        
        self.b_prime = tf.Variable(tf.truncated_normal([self.vocab_size], stddev=self.init_value * 0.001, mean=0), name='decoder_W', dtype=tf.float32 )
        self.params['b_prime'] = self.b_prime 
        
        self.encoded_values, self.decoded_values, self.masked_decoded_values, self.error, self.loss, self.train_step, self.summary = self.build_model()
        
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())
        self.log_writer = tf.summary.FileWriter(self.summary_path, graph = self.sess.graph)
        self._glo_ite_counter = 0
    
    def __del__( self ):
        if self.log_writer:
            self.log_writer.close()
            self.log_writer = None
    
    def build_model(self):     
        
        dense_masker = tf.sparse_tensor_to_dense(self.mask)
           
        with tf.name_scope('encoding'):
            encoding = tf.add(tf.sparse_tensor_dense_matmul(self.X, self.W) , self.b, name= 'raw_values')
            encoded_values = self.enc_func(encoding, name = 'encoded_values')
        
        with tf.name_scope('decoding'):
            decoding =  tf.nn.xw_plus_b(encoded_values, self.W_prime, self.b_prime)
            decoded_values = self.dec_func(decoding, name = 'decoded_values')
            masked_decoded_values = tf.multiply(dense_masker, decoded_values)
        
        with tf.name_scope('training_process'):
            diff = tf.squared_difference(tf.sparse_tensor_to_dense(self.Y, default_value = 0) , decoded_values)
            error  = tf.reduce_sum( tf.multiply(dense_masker, diff) )
            reg = 0  
            for param in self.params.items():
                reg += tf.nn.l2_loss(param[1])* self.lambda_w
            loss = error + reg
            
        model_params = [p for p in self.params.values()]
           
        train_step = self._optimize(loss, model_params)  
          
        tf.summary.scalar('error', error)
        tf.summary.scalar('loss', loss)        
        for param in self.params.items():
            tf.summary.histogram(param[0], param[1])   
        #tf.summary.histogram('predictions', decoded_values)     
        merged_summary = tf.summary.merge_all()
                       
        return encoded_values, decoded_values, masked_decoded_values, error, loss, train_step, merged_summary

    def _optimize(self, loss , model_params ):
        if self.opt == 'adadelta':
            train_step = tf.train.AdadeltaOptimizer(self.lr, self.rho, self.epsilon).minimize(loss, var_list= model_params)
        elif self.opt == 'sgd':
            train_step = tf.train.GradientDescentOptimizer(self.lr, self.beta1, self.beta2, self.epsilon).minimize(loss,var_list=model_params)
        elif self.opt =='adam':
            train_step = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=model_params)
        elif self.opt =='ftrl':
            train_step = tf.train.FtrlOptimizer(self.lr).minimize(loss,var_list=model_params)
        return train_step
  

    def fit(self, sp_indices, sp_noised_values, sp_original_values, sp_mask_indices, cur_batch_size):
        tensor_shape = np.array([cur_batch_size, self.vocab_size], dtype=np.int64)
        error, loss, summary, _ = self.sess.run(
                  [self.error, self.loss, self.summary, self.train_step], 
                  {self.X: (sp_indices, sp_noised_values, tensor_shape), 
                   self.Y: (sp_indices, sp_original_values, tensor_shape), 
                   self.mask: (sp_mask_indices, np.ones(sp_mask_indices.shape[0]), tensor_shape)}
        )
        self.log_writer.add_summary(summary, self._glo_ite_counter)
        self._glo_ite_counter+=1
        return (error, loss)

    def get_encoding(self, sp_indices, sp_noised_values , cur_batch_size):
        tensor_shape = np.array([cur_batch_size, self.vocab_size], dtype=np.int64)
        return self.sess.run(self.encoded_values,
                    {self.X: (sp_indices, sp_noised_values, tensor_shape)}
        )
        
    def get_predictions(self, sp_indices, sp_noised_values, sp_original_values, sp_mask_indices, cur_batch_size):
        tensor_shape = np.array([cur_batch_size, self.vocab_size], dtype=np.int64)
        return self.sess.run([self.decoded_values, self.masked_decoded_values],
                    {self.X: (sp_indices, sp_noised_values, tensor_shape),
                     self.mask: (sp_mask_indices, np.ones(sp_mask_indices.shape[0]), tensor_shape)}
        )
    
    def evaluate(self, sp_indices, sp_noised_values, sp_original_values, sp_mask_indices, cur_batch_size ):
        tensor_shape = np.array([cur_batch_size, self.vocab_size], dtype=np.int64)
        error, loss = self.sess.run([self.error, self.loss], 
                   {self.X: (sp_indices, sp_noised_values, tensor_shape), 
                   self.Y: (sp_indices, sp_original_values, tensor_shape), 
                   self.mask: (sp_mask_indices, np.ones(sp_mask_indices.shape[0]), tensor_shape)}
                  )
        return (error, loss)

   
    def get_activation_func(self, name):
        if name == "tanh":
            return tf.tanh 
        elif name == 'sigmoid':
            return tf.sigmoid
        elif name == 'identity':
            return tf.identity
        elif name == 'relu':
            return tf.nn.relu
        elif name == 'relu6':
            return tf.nn.relu6
        
        config.logger.info('unsupported activation type! %s'  %(name))
        return tf.tanh 
    
    def save_model(self, filename, step):
        if not self.saver:
            self.saver = tf.train.Saver(max_to_keep = 50)
        self.saver.save(self.sess, filename, global_step = step )
        
    def restore_model(self, filename):
        if not self.saver:
            self.saver = tf.train.Saver(max_to_keep = 50)
        self.saver.restore(self.sess, filename)
        