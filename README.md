# rnn_recsys
Our implementation of one research paper "Embedding-based News Recommendation for Millions of Users" https://dl.acm.org/citation.cfm?id=3098108 Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. 2017. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17). 


I provide a toy demo dataset to demonstrate the file format. On this dataset, model AVG has an AUC of 0.76, and model RNN has an AUC of 0.92.  You can reproduce this simply by running 'python train.py' . Sorry that I cannot upload my own real-world dataset (Bing News).

Overall, this recommender system has two steps: (1) train an autoencoder for articles ; (2) train RNN base on user-item interactions. 

## training autoencoder
The raw article has a format of "article_id \t category \t title". I will first build a word dictionary to hash the word to ids and count the TF-IDF statistics. The input for training autoencoder is the TF-IDF values of each article (title). Below is a result of my trained CDAE. (scripts can be found in helper/demo.py):
