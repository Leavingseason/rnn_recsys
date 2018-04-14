'''
@author: v-lianji
'''
import re
import pickle
from collections import Counter
import math
from operator import itemgetter

def try_get_param(hparams, key, dvalue):
    if key in hparams:
        return hparams[key] 
    else:
        return dvalue
    

def clean_str(string):
    """
    reuse the code from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def dump_obj_to_file(filename, obj):
    with open(filename, 'wb') as wt:
        pickle.dump(obj, wt, pickle.HIGHEST_PROTOCOL)

def load_obj_from_file(filename):
    with open(filename, 'rb') as rd:
        obj = pickle.load(rd)
    return obj

def convert_line_to_tfidf(line, wh_model, norm = False):
    IDF_CONSTANT = 1000000
    tokens = clean_str(line).split(' ')
    if len(tokens)<2:
        return None    
    cur_word_dict = Counter(tokens)
    cur_word_list = [(wh_model.word2idx[k],v) for k,v in cur_word_dict.items() if k in wh_model.word2idx]
    if not cur_word_list:
        return None
    cur_word_list.sort()
    doc_word_cnt = sum(p[1] for p in cur_word_list) * 1.0 
    if doc_word_cnt<=0.001:
        return None
    res = [(p[0],p[1]*1.0/doc_word_cnt * math.log2(IDF_CONSTANT*1.0/wh_model.word2freq[wh_model.idx2word[p[0]]])) for p in cur_word_list]
    if norm:
        max_value = max(0.001,max(res, key = itemgetter(1))[1])
        res = [(p[0],p[1]/max_value) for p in res]
    return res

def get_firstlines(infile, outfile, k):
    cnt = 0 
    with open(infile, 'r') as rd:
        with open(outfile, 'w') as wt:
            while True:
                line = rd.readline() 
                cnt += 1 
                if not line or cnt>k :
                    break 
                wt.write(line)

if __name__ == '__main__':
    # testing 
    s = "hello    world my're name's. (jianxun)! how \ndo you'll it isn't do? OK,you are not felling very well. the@exam is so "
    print(s)
    print(clean_str(s))