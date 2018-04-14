'''
@author: v-lianji
'''

import codecs 
from collections import defaultdict, OrderedDict
import operator 
import pickle 
import config
from helper import utils
import random 

class word_hash_agent(object):
    
    def __init__(self, topk = 100000):
        self.topk_words = topk
    
    def clear(self):
        self.word2idx = {} 
        self.idx2word = {}
        
        self.word2freq = defaultdict(int)
        
    def do_hashing(self, infile):
        self.clear()
        linecnt = 0 
        with codecs.open(infile, 'r', 'utf-8') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break 
                linecnt+=1
                words = line.strip().split('\t')
                #if len(words)!=2 or not line.startswith('http'):
                if len(words)!=3:
                    continue
                tokens = set(utils.clean_str(words[2]).split(' '))
                for token in tokens:
                    self.word2freq[token]+=1
                    
        raw_word_cnt = len(self.word2freq)
        
   
        config.logger.info('total raw words: %d, number of lines: %d' %(raw_word_cnt, linecnt))
        
        self.word2freq = self.select_top_words(self.word2freq, self.topk_words)
        
        r'''
        idx= 0 
        for p in self.word2freq.items():
            self.word2idx[p[0]]=idx 
            self.idx2word[idx]=p[0]
            idx+=1        
        '''
        
        '''
        shuffle the words for better indexing
        '''
        words_bag = list(self.word2freq.keys())
        random.shuffle(words_bag)
        idx= 0 
        for p in words_bag:
            self.word2idx[p]=idx 
            self.idx2word[idx]=p
            idx+=1
        
        config.logger.info('finished word hashing.')
            
    def select_top_words(self, word2freq, k):
        pairs = sorted(self.word2freq.items(), key = operator.itemgetter(1), reverse = True)  
        print(pairs[0:10])
        res = OrderedDict()
        for p in pairs[0:min(k,len(pairs))]:
            res[p[0]]=p[1]
            
        return res
    

        