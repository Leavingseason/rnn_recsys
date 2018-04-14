'''
@author: v-lianji
'''
import config 
import tensorflow as tf 
import numpy as np 
import math
from models.DAE import SDAE


class CDAE(SDAE):
    '''
    category-constraint DAE, as described in 'Embedding-based News Recommendation for Millions of Users', section 3.1 
    inner product of two encoding vectors is larger if the two documents belong to the same category
    '''

    def __init__(self, **hparam):
        '''
        Constructor
        '''
        
        self.alpha_enc = hparam['alpha_enc'] if 'alpha_enc' in hparam else 0.1
        self.X1 = tf.sparse_placeholder(tf.float32) 
        self.Y1 = tf.sparse_placeholder(tf.float32) 
        self.mask1 = tf.sparse_placeholder(tf.float32) 

        self.X2 = tf.sparse_placeholder(tf.float32) 
        self.Y2 = tf.sparse_placeholder(tf.float32) 
        self.mask2 = tf.sparse_placeholder(tf.float32) 
        
        config.logger.info(str(hparam))

        super().__init__(**hparam)
        
    def build_model(self):    
        
        dense_masker01 = tf.sparse_tensor_to_dense(self.mask)
        dense_masker02 = tf.sparse_tensor_to_dense(self.mask1)
        dense_masker03 = tf.sparse_tensor_to_dense(self.mask2)
            
        with tf.name_scope('encoding'):
            encoding = tf.add(tf.sparse_tensor_dense_matmul(self.X, self.W) , self.b, name= 'raw_values')
            encoded_values = self.enc_func(encoding, name = 'encoded_values') - self.enc_func(self.b)  
            
            encoding1 = tf.add(tf.sparse_tensor_dense_matmul(self.X1, self.W) , self.b, name= 'raw_values1')
            encoded_values1 = self.enc_func(encoding1, name = 'encoded_values1') - self.enc_func(self.b)  
            
            encoding2 = tf.add(tf.sparse_tensor_dense_matmul(self.X2, self.W) , self.b, name= 'raw_values2')
            encoded_values2 = self.enc_func(encoding2, name = 'encoded_values2') - self.enc_func(self.b)  
    
            
        with tf.name_scope('decoding'):
            decoding =  tf.nn.xw_plus_b(encoded_values, self.W_prime, self.b_prime)
            decoded_values = self.dec_func(decoding, name = 'decoded_values')
            
            decoding1 =  tf.nn.xw_plus_b(encoded_values1, self.W_prime, self.b_prime)
            decoded_values1 = self.dec_func(decoding1, name = 'decoded_values1')
            
            decoding2 =  tf.nn.xw_plus_b(encoded_values2, self.W_prime, self.b_prime)
            decoded_values2 = self.dec_func(decoding2, name = 'decoded_values2')
            
            masked_decoded_values = tf.multiply(dense_masker01, decoded_values)
        
        with tf.name_scope('training_process'):
            diff01 = tf.squared_difference(tf.sparse_tensor_to_dense(self.Y) , decoded_values)  
            diff02 = tf.squared_difference(tf.sparse_tensor_to_dense(self.Y1) , decoded_values1)  
            diff03 = tf.squared_difference(tf.sparse_tensor_to_dense(self.Y2) , decoded_values2)
            L_R  = tf.reduce_sum( tf.multiply(dense_masker01, diff01)) \
                +  tf.reduce_sum( tf.multiply(dense_masker02, diff02)) \
                + tf.reduce_sum( tf.multiply(dense_masker03, diff03))
            
            L_T = tf.reduce_sum( tf.log(1+ tf.exp( tf.reduce_sum( tf.multiply(encoded_values, encoded_values2), 1) -  tf.reduce_sum(tf.multiply(encoded_values, encoded_values1),1))))
            
            error = L_R + self.alpha_enc * L_T
            
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
        merged_summary = tf.summary.merge_all()                                   
            
        return encoded_values, decoded_values, masked_decoded_values, error, loss, train_step, merged_summary
    
    def fit(self,  sp_indices, sp_noised_values, sp_original_values, sp_mask_indices,
            sp_indices1, sp_noised_values1, sp_original_values1,sp_mask_indices1,
            sp_indices2, sp_noised_values2, sp_original_values2, sp_mask_indices2, cur_batch_size 
            ):
        
        tensor_shape = np.array([cur_batch_size, self.vocab_size], dtype=np.int64)
        
        error, loss, summary, _ = self.sess.run(
                  [self.error, self.loss, self.summary, self.train_step], 
                  {self.X: (sp_indices, sp_noised_values, tensor_shape), 
                   self.Y: (sp_indices, sp_original_values, tensor_shape), 
                   self.mask: (sp_mask_indices, np.ones(sp_mask_indices.shape[0]), tensor_shape),
                   
                   self.X1: (sp_indices1, sp_noised_values1, tensor_shape), 
                   self.Y1: (sp_indices1, sp_original_values1, tensor_shape), 
                   self.mask1: (sp_mask_indices1, np.ones(sp_mask_indices1.shape[0]), tensor_shape),
                   
                   self.X2: (sp_indices2, sp_noised_values2, tensor_shape), 
                   self.Y2: (sp_indices2, sp_original_values2, tensor_shape), 
                   self.mask2: (sp_mask_indices2, np.ones(sp_mask_indices2.shape[0]), tensor_shape)
                   }
        )
        self.log_writer.add_summary(summary, self._glo_ite_counter)
        self._glo_ite_counter+=1
        return (error, loss)
