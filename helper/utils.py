'''
@author: v-lianji
'''
import re
import pickle

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

if __name__ == '__main__':
    # testing 
    s = "hello    world my're name's. (jianxun)! how \ndo you'll it isn't do? OK,you are not felling very well. the@exam is so "
    print(s)
    print(clean_str(s))