# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# str = lemmatizer.lemmatize('were')
# print("The word2 is:", str)

# from nltk import word_tokenize, pos_tag
# from nltk.corpus import wordnet
# from nltk.stem import WordNetLemmatizer
# import nltk
# import re
# from nltk.corpus import stopwords
#
#
# from nltk import word_tokenize, pos_tag
# from nltk.corpus import wordnet
# from nltk.stem import WordNetLemmatizer
#
# def clean_sen(string):#清理句子
#     string = re.sub(r"[^A-Za-z0-9\'.,\-]", " ", string)#余留小数点.的意思是为了保存出现的小数；保留逗号，的意思是保存300，000这样的大树；保留引号‘的意思是保留缩写
#
#     # string = re.sub(r"\'s", "\'s", string)
#     # string = re.sub(r"n\'t", "n\'t", string)
#     # string = re.sub(r"\'ll", "\'ll", string)
#     #
#     # string = re.sub(r"\'ve", "\'ve", string)
#     # string = re.sub(r"\'re", "\'re", string)
#     # string = re.sub(r"\'d", "\'d", string)
#     #
#     string = re.sub(r"\s{2,}", " ", string)#将多余的空格合并为一个
#
#     return string.strip()
#     # return string.strip().lower()#全部转为小写
#
#
#
# # 判断该单词是否需要还原
# def is_need_lemmas(tag):#都是需要还原的
#     if tag.startswith('J'):
#         return 1
#     elif tag.startswith('V'):
#         return 1
#     elif tag.startswith('N'):
#         return 1
#     elif tag.startswith('R'):
#         return 1
#     else:
#         return 0
#
#
#
# # 获取需要还原单词的词性
# def get_wordnet_pos(tag):#都是需要还原的
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV


# sentence = 'football was a family of team sports that involve, to varying degrees, kicking a ball to score a goal, plays, playing, has, had, have, were, did, does, do, are, were'
# sentence = 'have had has do does did doing be is are were was play playing plays played can could $19.5 million'
# sentence = 'I dont knew who (Jonathan Gruber) was.'
# sentence = 'Sen. Obama\'s campaign announced that he\'s choosing his cabinet 20.'
# # sentence = 'doesn\'t don\'t she\'ll she\'s '
# sentence = 'Transgender individuals in the U.S. have a 1-in-12 chance of being murdered.'
# sentence = 'Raising the minimum "wage" to $10.10 300,000 an hour, would co-response help lift over ？ ￥ ! -- (a million Americans out) of poverty she\'s.'
# sentence = 'ISIS supporter tweeted at 10:34 a.m. Shooting began at 10:45 a.m. in Chattanooga, as an actor.'
# sentence = 'a pleasing , often-funny comedy . '
#
# tokens = word_tokenize(sentence)  # 分词
# # print("The tokens is:", tokens)
# tagged_sent = pos_tag(tokens)     # 获取单词词性
# # print("The tagged_sent is:", tagged_sent)
#
# wnl = WordNetLemmatizer()
# lemmas_sent = []
# for tag in tagged_sent:
#     # print("The tag is:",tag)
#     islemmas_sent = is_need_lemmas(tag[1])
#     if islemmas_sent:
#        # print("1")
#        wordnet_pos = get_wordnet_pos(tag[1])
#        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
#     else:
#         # print("2")
#         lemmas_sent.append(tag[0])
#
# print(lemmas_sent)
#
# from stanfordcorenlp import StanfordCoreNLP
# import nltk
# from nltk.tree import Tree as nltkTree
#
##读取stanford-corenlp所在的目录
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'.\stanford-corenlp-full-2016-10-31', lang='en')
sentence = 'I like playing basketball'
res = nlp.dependency_parse(sentence)
print(res)

# lemmas_sent1 = []
# # 输入句子
# sentence = ' . . . a visually seductive , unrepentantly trashy take on rice\'s second installment of her vampire chronicles . '
# words = nlp.word_tokenize(sentence)
# doc_words = []
# for word in words:
#     print(word)
#     if word != "," and word != ":" and word != "."and word != "'"and word != "--":
#         doc_words.append(word)
# doc_str = ' '.join(doc_words).strip()
# print(doc_str)

# # print('Part of Speech:', nlp.pos_tag(sentence))
# tagged_sent = nlp.pos_tag(sentence)
# for tag in tagged_sent:
#     # print("The tag is:",tag)
#     islemmas_sent = is_need_lemmas(tag[1])
#     if islemmas_sent:
#        # print("1")
#        wordnet_pos = get_wordnet_pos(tag[1])
#        lemmas_sent1.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
#     else:
#         # print("2")
#         lemmas_sent1.append(tag[0])
#
# print(lemmas_sent)
# print(lemmas_sent1)


# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
# print("The len is:", len(stop_words))
# print("The english stop_words is:")
# print(stop_words)

# -*- coding: UTF-8 -*-

# Filename : test.py
# author by : www.runoob.com

# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         pass
#
#     try:
#         import unicodedata
#         unicodedata.numeric(s)
#         return True
#     except (TypeError, ValueError):
#         pass
#
#     return False
#
#
# # 测试字符串和数字
# print(is_number('foo'))  # False
# print(is_number('1'))  # True
# print(is_number('1.3'))  # True
# print(is_number('-1.37'))  # True
# print(is_number('1e3'))  # True
# print(is_number('10.10'))  # True
# print(is_number('300,000'))  # True
#
# # 测试 Unicode
# # 阿拉伯语 5
# print(is_number('٥'))  # True
# # 泰语 2
# print(is_number('๒'))  # True
# # 中文数字
# print(is_number('四'))  # True
# # 版权号
# print(is_number('©'))  # False

# from nltk.corpus import stopwords
# import nltk
#
# # nltk.download('stopwords')
# # stop_words = set(stopwords.words('english'))
# # print("The len is:", len(stop_words))
# # print("The english stop_words is:")
# # print(stop_words)
# #
# # a = []
# # a.append(1)
# # print(a)
# # print(len(a))
# # a.append(2)
# # print(a)
# # print(len(a))
# # a.append(1)
# # print(a)
# # print(len(a))
# #
# # a = set()
# # a.add(1)
# # print(a)
# # print(len(a))
# # a.add(2)
# # print(a)
# # print(len(a))
# # a.add(1)
# # print(a)
# # print(len(a))
#
#
# # a = 1
# # b =[0, 1]
# # if a in b:
# #     print(1)
# #
# # import numpy as np
# # def sample_mask(idx, l):
# #     #调用的函数为：train_mask = sample_mask(idx_train, labels.shape[0])
# #     """Create mask."""
# #     mask = np.zeros(l)
# #     mask[idx] = 1
# #     return np.array(mask, dtype=np.bool)
# #
# # idx_train = range(2)
# # idx_val = range(2,3)
# # idx_test = range(3,4)
# #
# # train_mask = sample_mask(idx_train, 4)#sample_mask用到了上面定义的函数，labels.shape[0]指的是整个语料库中共有多少个标签，即多少条语句，即多少行，然后分别给训练集、验证集、测试集来标注上，以区分开
# # val_mask = sample_mask(idx_val, 4)
# # test_mask = sample_mask(idx_test, 4)
# #
# # print(train_mask)
# # print(val_mask)
# # print(test_mask)
# #
# # ally = [[0,1],
# #         [1,0],
# #         [0,1]]
# # ty = [[0,1]]
# # labels = np.vstack((ally, ty))
# # print(labels)
# # print(labels.shape)
# #
# # y_train = np.zeros(labels.shape)
# # y_val = np.zeros(labels.shape)
# # y_test = np.zeros(labels.shape)
# # print(y_train)
# # print(y_val)
# # print(y_test)
# #
# # y_train[train_mask, :] = labels[train_mask, :]
# # y_val[val_mask, :] = labels[val_mask, :]
# # y_test[test_mask, :] = labels[test_mask, :]
# # print(y_train)
# # print(y_val)
# # print(y_test)
#
# # import numpy as np
# # from scipy.sparse import coo_matrix
# #
# # adj = coo_matrix((np.ones(5), ([3, 4, 0, 2, 1], [0, 2, 1, 4, 3])), shape=(5, 5), dtype=np.float32)
# # print(adj)
# # print()
# # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# # print(adj)
#
# # import numpy as np
# # from tqdm import tqdm
# # import scipy.sparse as sp
# #
# #
# # def sparse_to_tuple(sparse_mx):  # 将矩阵转换成tuple格式并返回
# #     """Convert sparse matrix to tuple representation."""
# #
# #     def to_tuple(mx):
# #         if not sp.isspmatrix_coo(mx):
# #             mx = mx.tocoo()
# #         coords = np.vstack((mx.row, mx.col)).transpose()
# #         values = mx.data
# #         shape = mx.shape
# #         return coords, values, shape
# #
# #     if isinstance(sparse_mx, list):
# #         for i in range(len(sparse_mx)):
# #             sparse_mx[i] = to_tuple(sparse_mx[i])
# #     else:
# #         sparse_mx = to_tuple(sparse_mx)
# #
# #     return sparse_mx
# #
# #
# # def normalize_adj(adj):  # 图归一化并返回，归一化：把数据映射到0～1范围之内处理，更加便捷快速
# #     """Symmetrically normalize adjacency matrix."""
# #     rowsum = np.array(adj.sum(1))
# #     with np.errstate(divide='ignore'):
# #         d_inv_sqrt = np.power(rowsum, -0.5).flatten()
# #     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
# #     d_mat_inv_sqrt = np.diag(d_inv_sqrt)
# #     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
# #
# #
# # def preprocess_adj(adj):  # 处理得到GCN中的归一化矩阵并返回
# #     """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
# #     max_length = max([a.shape[0] for a in adj])
# #     mask = np.zeros((adj.shape[0], max_length, 1))  # mask for padding
# #
# #     for i in tqdm(range(adj.shape[0])):
# #         adj_normalized = normalize_adj(adj[i])  # no self-loop#将原来的矩阵归一化，矩阵中的每一个元素都转化为0-1直接的数
# #
# #         pad = max_length - adj_normalized.shape[0]
# #         adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)),
# #                                 mode='constant')  # 目的就是填充，使得每一个句子的矩阵都具有相同的形状，因为句子的长短不一，也就造成初始的每个矩阵形状不统一，这个目的就是
# #         # 使得每个句子的矩阵都具有相同的形状
# #         mask[i, :adj[i].shape[0], :] = 1.
# #         adj[i] = adj_normalized
# #     # return adj, mask
# #     return sparse_to_tuple(adj), mask
# #
# #
# # adj = []
# # a = [[2, 1],
# #      [1, 2]]
# # a = np.array(a)
# # adj.append(a)
# # b = [[1, 2, 3],
# #      [2, 1, 3],
# #      [1, 1, 3]]
# # b = np.array(b)
# # adj.append(b)
# #
# # c = [[1]]
# # c = np.array(c)
# # adj.append(c)
# #
# # adj = np.array(adj)
# #
# # print(adj)
# #
# # train_adj, train_mask = preprocess_adj(adj)
# # print(train_adj)
# # print()
# # print(train_mask)
#
# import numpy as np
# # adj = []
# # m1 = np.matrix([[1, 2], [2, 1]])
# # print(type(m1))
# # adj.append(adj)
# # m2 = np.matrix([[1, 3], [2, 1]])
# # adj.append(adj)
# #
# # print(type(adj))
# #
# # adj = np.mat(adj)
# # print(type(adj))
#
# # 1、list变array：np.array(list)
# #
# # 2、array变list：data.tolist()
# #
# # 3、array与matrix相互转化：np.asmatrix和np.asarray()
# #
# # 4、list变matrix：np.mat(list)
#
# # import random
# #
# # initial = random.random()
# #
# # print(initial)
#
#
# import numpy as np
# a = np.ones((3))
# print(a)
# print(type(a))
# print(a*2)
#
# import tensorflow as tf
#
# a = tf.ones((2,2))
# print(a)
# print(type(a))
# print(a*2)


