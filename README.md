# rnn_recsys
Our implementation of one research paper "Embedding-based News Recommendation for Millions of Users" https://dl.acm.org/citation.cfm?id=3098108 Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. 2017. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17). 


I provide a toy demo dataset to demonstrate the file format. On this dataset, model AVG has an AUC of 0.76, and model RNN has an AUC of 0.92.  You can reproduce this simply by running 'python train.py' . Sorry that I cannot upload my own real-world dataset (Bing News).

Overall, this recommender system has two steps: (1) train an autoencoder for articles ; (2) train RNN base on user-item interactions. 

## training autoencoder
The raw article has a format of "article_id \t category \t title". I will first build a word dictionary to hash the word to ids and count the TF-IDF statistics. The input for training autoencoder is the TF-IDF values of each article (title). Below is a result of my trained CDAE. (scripts can be found in helper/demo.py):
![alt text](https://github.com/Leavingseason/rnn_recsys/blob/master/notes/CDAE/demo2.JPG)
Analysis: I am really surprised by the great power of autoencoder. News titles are usually very short, but autoencoder can recover their intent. For example, for the first news, the input content is 'Travel tips for thanksgiving',  our decoded words are (ordered by importance) 'tips, travel, holidays, thanksgiving, enjoy, shopping, period'. Note that the words 'holidays' and 'shopping' do not appear in the original title, but there are captured as strongly related words.

training curve of error:

![alt text](https://github.com/Leavingseason/rnn_recsys/blob/master/notes/CDAE/error2.JPG)

training curve of loss:

![alt text](https://github.com/Leavingseason/rnn_recsys/blob/master/notes/CDAE/loss2.JPG)


## training autoencoder
After training the autoencoder, you need to encode each raw article to get their embeddings:

encode_articles(...)

Finally, train your RNN recsys:

train_RS()

## data description:
data/articles.txt:  each line is an article, in the form of word_id:word_tf_idf_value 

data/articles_CDAE.txt: each line is 3 articles, splited by tab, article_01 and article_02 belong to a same category, while article_03 belongs to a different category 

data/RS/articles_embeddings.txt:  each line is an article,  in the form of article_ID \t embedding_vector (D float numbers, where D denotes the dimension of embedding) 

data/RS/train.txt:  each line is a training instance, in the form of user_history \t target_item_id \t label.   user_history is a sequence of item_id, splited by space. 

articles_TFIDF_norm_3w.txt format: each line is one document, such as 14 17513:1.00 27510:0.81  , representing  article_id \TAB word_id01:value \ SPACE word_id02:value \SPACE ....
