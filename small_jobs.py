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
    '''
    [('rt_china', 6.967694285445531e-07), ('rt_france', 6.967694285445531e-07), ('rt_germany', 6.967694285445531e-07), ('rt_arts', 6.967694285445531e-07), 
    ('rt_taiwan', 6.967694285445531e-07), ('rt_brazil', 2.0903082856336594e-06), ('rt_singapore', 2.020631342779204e-05), ('rt_factcheck', 2.578046885614847e-05), 
    ('rt_indonesia', 4.2502935141217745e-05), ('rt_obituary', 6.619309571173255e-05), ('rt_ireland', 9.615418113914833e-05), ('rt_sports_nhra', 0.00011427018628130672), 
    ('rt_newzealand', 0.0001282055748521978), ('rt_sports_indycar', 0.0001351732691376433), ('rt_sports_esports', 0.0004368744316974348), 
    ('rt_southafrica', 0.0007023435839729097), ('rt_sports_softball', 0.0007838656071126223), ('rt_australia', 0.0010430638345311962), 
    ('rt_sports_horseracing', 0.0010799926142440574), ('rt_india', 0.0013252554530917401), ('rt_sports_cycling', 0.0013259522225202848), 
    ('rt_sports_wrestling', 0.0014834221133713536), ('rt_uk', 0.0017816394287884225), ('rt_products', 0.002028295806493194), ('rt_sports_nascar', 0.0023272098913388077),
     ('rt_sports_wwe', 0.0023348743550527977), ('rt_sports_boxing', 0.002418486686478144), ('rt_sports_tennis', 0.0031939910604482317),
      ('rt_sports_cricket', 0.00336051895387038), ('rt_world_americas', 0.0037618581447120426), ('rt_sports_golf', 0.0039897017478461114), 
      ('rt_canada', 0.004174345646410418), ('rt_sports_mma', 0.004416124638115378), ('rt_sports_cbb', 0.0056096906692121975), ('rt_world_middleeast', 0.005647316218353604), 
      ('rt_science', 0.005875159821487673), ('rt_world_asia', 0.0059977912409115135), ('rt_sports_nhl', 0.006256292698901543), ('rt_world_europe', 0.006407491664895711), 
      ('rt_us_midwest', 0.006530819853748097), ('rt_us_northeast', 0.0066297611126014235), ('rt_entertainment_music', 0.006773992384310146), 
      ('rt_sports_mlb', 0.0067997728531662945), ('rt_world_africa', 0.007104957862868809), ('rt_entertainment_movieandtv', 0.00785955915398256), 
      ('rt_sports_nba', 0.008932584073941171), ('rt_us_west', 0.008986235319939102), ('rt_sports_soccer', 0.009724810914196328), ('rt_sports_cfb', 0.011035434209288634), 
      ('rt_us_south', 0.011492514954413861), ('rt_health', 0.012670752058082699), ('rt_technology', 0.013154310041492619), ('rt_world', 0.016245875995944802), 
      ('rt_sports_nfl', 0.019287274551541778), ('rt_business', 0.022477781764847285), ('rt_entertainment', 0.03942739488362208), ('rt_scienceandtechnology', 0.04716989677360916),
       ('rt_politics', 0.06248001142701863), ('rt_sports', 0.08717142966635195), ('rt_us', 0.09977459508986583), ('null', 0.10282156780089116), 
       ('rt_lifestyle', 0.12377481805608297), ('rt_unclassified', 0.18327823048435926)]
    '''
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
    
    r'''
    convert_raw_file_to_indexed(
        r'D:\My Projects\data\news\artitle_title_2017-01-01-2017-01-31.firstlines.tsv'  ,
        r'D:\My Projects\data\news\processed\artitle_title_2017-01-01-2017-01-31_word_index.txt'   ,
        r'D:\My Projects\data\news\processed\word_hashing.obj'                   
                                )
        '''