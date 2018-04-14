'''
@author: v-lianji
'''

import config
from helper import *
import codecs  
from helper import utils, data_loader
from collections import Counter, defaultdict
import math 
import random 
import numpy as np 
import operator


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

def prepare_autoencoder_files(infile, outfile ):
    with codecs.open(infile, 'r', 'utf-8') as rd:
        with codecs.open(outfile, 'w', 'utf-8') as wt:
            while True:
                line = rd.readline()
                if not line:
                    break  
                words = line.split('\t')
                wt.write(words[1])

def prepare_triple_autoencoder_files(article_raw, article_IFIDF, outfile):
    '''
    step 1 : from article_raw load a list of (article id, category)
    step 2 : shuffle the list 
    step 3 : split the list into two disjoint list: A and B
    step 4 : build a reverted-index C from category to article list on B 
    step 5 : for each article in A, find an article from C which has the same category, and find an article from C which has a different catetory
    '''
    all_articles = load_article_category_as_list(article_raw)
    random.shuffle(all_articles)
    articles_parta, articles_partb = split_list(all_articles)
    category_list, category_prob, cate2articles, cate2cnt = build_category_probability_and_inverted_index(articles_partb) 
    print(cate2cnt)
    debug_list = [(a,b) for a,b in zip(category_list,category_prob)]
    debug_list .sort(key = operator.itemgetter(1))
    print(debug_list)
    article2content =  data_loader.load_article_content(article_IFIDF)
    
    with open(outfile, 'w') as wt:
        for article in articles_parta:
            if article[0] not in article2content or article[1] not in cate2cnt:
                continue
            selected_category = article[1]
            while selected_category==article[1]:
                selected_category = np.random.choice(category_list, size=1, replace=True, p = category_prob)[0]
            article_same_cate = sample_one_article(cate2articles[article[1]], article2content)
            article_diff_cate = sample_one_article(cate2articles[selected_category], article2content)
            wt.write('{0}\t{1}\t{2}\n'.format(article2content[article[0]], article2content[article_same_cate], article2content[article_diff_cate]))
     

def sample_one_article(mylist, d):
    res =  random.choice(mylist) 
    while res not in d:
        res =  random.choice(mylist) 
    return res

def split_list(mylist):
    cnt = len(mylist) 
    mid = cnt//2 
    return mylist[0:mid], mylist[mid:]

def build_category_probability_and_inverted_index(mylist):
    cate2cnt = defaultdict(int)
    cate2articles = defaultdict(list)
     
    for p in mylist:
        cate2cnt[p[1]]+=1   
        cate2articles[p[1]].append(p[0])
    
    cate2cnt = dict([(p[0],p[1]) for p in cate2cnt.items() if p[1]>1000 and not (p[0] == 'null' or p[0] == 'rt_unclassified')])   
        
    total = sum([p[1] for p in cate2cnt.items()])
    cate_list, cate_prob = [] , [] 
    for p in cate2cnt.items():
        cate_list.append(p[0])
        cate_prob.append(p[1]*1.0/total) 
        
    return cate_list, cate_prob, cate2articles, cate2cnt
             
def load_article_category_as_list(infile):

    articles = [] 
    with codecs.open(infile, 'r', 'utf-8') as rd:
        while True:
            line = rd.readline() 
            if not line:
                break 
            words = line.strip().split('\t')
            if len(words)!=3:
                continue 
            articles.append((words[0], words[1].lower()))  
    return articles           
                
def convert_raw_file_to_indexed(infile,outfile,word_hashing_file,norm=False):
    r'''
    input format: id\t category\t title
    output format: id\t word:weight ...
    '''
    wh_model = utils.load_obj_from_file(word_hashing_file)
    with codecs.open(infile, 'r', 'utf-8') as rd:
        with codecs.open(outfile, 'w', 'utf-8') as wt:
            while True:
                line = rd.readline()
                if not line:
                    break  
                words = line.strip().split('\t')
                #if len(words)!=2 or not line.startswith('http'):
                if len(words)!=3 :# or not line.startswith('http'):
                    continue
                
                r'''
                tokens = utils.clean_str(words[2]).split(' ')
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
                wt.write(words[0]+'\t')
                for p in cur_word_list:
                    wt.write('{0}:{1:.2f} '.format(p[0],p[1]*1.0/doc_word_cnt * math.log2(1000000*1.0/wh_model.word2freq[wh_model.idx2word[p[0]]])) )
                wt.write('\n')
                ''' 
               
                cur_word_list = utils.convert_line_to_tfidf(words[2], wh_model, norm)
                if not cur_word_list:
                    continue 
                wt.write(words[0]+'\t')
                for p in cur_word_list:
                    wt.write('{0}:{1:.2f} '.format(p[0],p[1]))
                wt.write('\n')
                    

if __name__ == '__main__':
    
    r'''
    gen_word_hashing(r'D:\My Projects\data\news\artitle_title_2017-01-01-2017-01-31.tsv',
                     r'D:\My Projects\data\news\processed\word_hashing_title_top1w.obj', 10000)
    '''
    
    #load_wordhash_data(r'D:\My Projects\data\news\processed\word_hashing_title_2017-01-01-2017-01-31.obj')
 