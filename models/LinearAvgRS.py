'''
@author: v-lianji
'''
import tensorflow as tf 
import numpy as np 
import math
import config


class BaseRS(object):
    def __init__(self, **hparam ):
        
        self.init_value = hparam['init_value'] if 'init_value' in hparam else 0.01
        self.lambda_w = hparam['lambda_w'] if 'lambda_w' in hparam else 0.001 
        self.lr = hparam['learning_rate'] if 'learning_rate' in hparam else 0.001 
        self.opt = hparam['type_of_opt'] if 'type_of_opt' in hparam else 'adam'
        self.rho = hparam['rho'] if 'rho' in hparam else 0.95
        self.epsilon = hparam['epsilon'] if 'epsilon' in hparam else 1e-8
        self.beta1 = hparam['beta1'] if 'beta1' in hparam else 0.9
        self.beta2 = hparam['beta2'] if 'beta2' in hparam else 0.999
        
        self.summary_path = hparam['tf_summary_file'] if 'tf_summary_file' in hparam else 'log_tmp_path'
        
        self.type_of_loss = hparam['loss'] if 'loss' in hparam else 'rmse' 
        
        self.params = {} 
    
    def __del__( self ):
        if self.log_writer:
            self.log_writer.close()
            self.log_writer = None
    
    def save_model(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)
        
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
    
      
    def _get_activation_func(self, name):
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

    def _init_graph(self):
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())
        self.log_writer = tf.summary.FileWriter(self.summary_path, graph = self.sess.graph)
        self._glo_ite_counter = 0

        
class LinearAvgRS(BaseRS):
    '''
    linear average model, as described in paper 'Embedding-based News Recommendation for Millions of Users', section 4.3
    currently we didn't implement the decaying logic since the experimental results are not good,  as noted in the paper.
    '''

    def __init__(self, **hparam ):
        '''
        params:
             dim :  actually, the embedding size of document
        '''
                
        super().__init__(**hparam )
        
        self.dim = hparam['dim'] if 'dim' in hparam else 64 
        
        self.W = tf.Variable(
                            tf.truncated_normal([self.dim, 1], stddev=self.init_value, mean=0), 
                            name='W' , dtype=tf.float32
                            ) 
        self.params['W'] = self.W 
        
        self.X = tf.placeholder(tf.float32, shape=(None, self.dim))
        self.Y = tf.placeholder(tf.float32, shape=(None,1))
        
        self.predictions, self.error, self.loss, self.train_step, self.summary  = self._build_model()
        
        self._init_graph()

    def _build_model(self):
        with tf.name_scope('linear_regression'):
            preds = tf.matmul(self.X, self.W)
   
            if self.type_of_loss == 'cross_entropy_loss':
                error = tf.reduce_mean(
                               tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(preds, [-1]), labels=tf.reshape(self.Y, [-1])),
                               name='cross_entropy_loss'
                               )
            elif self.type_of_loss == 'square_loss' or self.type_of_loss == 'rmse':
                error = tf.reduce_mean(tf.squared_difference(preds, self.Y, name='squared_diff'), name='mean_squared_diff')
            elif self.type_of_loss == 'log_loss':
                error = tf.reduce_mean(tf.losses.log_loss(predictions=preds, labels=self.Y), name='mean_log_loss')
        
            reg = 0  
            for param in self.params.items():
                reg += tf.nn.l2_loss(param[1])* self.lambda_w
            loss = error + reg
            
        model_params = [p for p in self.params.values()] 
        train_step = self._optimize(loss, model_params)  
        
        tf.summary.scalar('error', error)
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('W', self.W)
        
        merged_summary = tf.summary.merge_all()
            
        return preds, error, loss, train_step, merged_summary
    
    def fit(self, X, Y, batch_size):
        error, loss, summary, _ = self.sess.run(
                  [self.error, self.loss, self.summary, self.train_step], 
                  {self.X: X, 
                   self.Y: Y
                   }
        )
        self.log_writer.add_summary(summary, self._glo_ite_counter)
        self._glo_ite_counter+=1
        return (error, loss)

    def pred(self, X, Y = None, batch_size = None):    
        '''
        return a numpy list
        '''
        preds = self.sess.run(
                  self.predictions, 
                  {self.X: X
                   }
        )
        return preds.reshape(-1)
    
    def evaluate(self, X, Y, batch_size):
        error, loss, preds  = self.sess.run(
                  [self.error, self.loss, self.predictions], 
                  {self.X: X, 
                   self.Y: Y
                   }
        )
        return (error, loss, np.reshape(preds, (-1)), np.reshape(Y, (-1)))
        
