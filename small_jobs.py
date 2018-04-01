'''
@author: v-lianji
'''

import config
from helper import *
import codecs  
from helper import utils
from collections import Counter
import math 


def gen_word_hashing(raw_file, outfile, topk ):  
    wh_model = word_hashing.word_hash_agent(topk) 
    wh_model.do_hashing(raw_file)    
    utils.dump_obj_to_file(outfile, wh_model)
    
def load_wordhash_data(filename):
    wh_model = utils.load_obj_from_file(filename)  
    cnt = 0 
    for p in wh_model.word2freq.items():
        print(p) 
        cnt+=1 
        if cnt>10:
            break   

def convert_raw_file_to_indexed(infile,outfile,word_hashing_file):
    wh_model = utils.load_obj_from_file(word_hashing_file)
    with codecs.open(infile, 'r', 'utf-8') as rd:
        with codecs.open(outfile, 'w', 'utf-8') as wt:
            while True:
                line = rd.readline()
                if not line:
                    break  
                words = line.strip().split('\t')
                if len(words)!=2 or not line.startswith('http'):
                    continue
                tokens = utils.clean_str(words[1]).split(' ')
                if len(tokens)<2:
                    continue
                cur_word_dict = Counter(tokens)
                cur_word_list = [(wh_model.word2idx[k],v) for k,v in cur_word_dict.items() if k in wh_model.word2idx]
                if not cur_word_list:
                    continue
                cur_word_list.sort()
                doc_word_cnt = sum(p[1] for p in cur_word_list) * 1.0 
                if doc_word_cnt<=0.001:
                    continue
                for p in cur_word_list:
                    wt.write('{0}:{1:.2f} '.format(p[0],p[1]*1.0/doc_word_cnt * math.log2(100000*1.0/wh_model.word2freq[wh_model.idx2word[p[0]]])) )
                wt.write('\n')

if __name__ == '__main__':
    

    gen_word_hashing(r'D:\My Projects\data\news\artitle_title_2017-01-01-2017-01-31.tsv',
                     r'D:\My Projects\data\news\processed\word_hashing_title_top1w.obj', 10000)

    
    #load_wordhash_data(r'D:\My Projects\data\news\processed\word_hashing_title_2017-01-01-2017-01-31.obj')
    
    r'''
    convert_raw_file_to_indexed(
        r'D:\My Projects\data\news\artitle_title_2017-01-01-2017-01-31.firstlines.tsv'  ,
        r'D:\My Projects\data\news\processed\artitle_title_2017-01-01-2017-01-31_word_index.txt'   ,
        r'D:\My Projects\data\news\processed\word_hashing.obj'                   
                                )
        '''