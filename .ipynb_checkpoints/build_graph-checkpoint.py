# 构建图


import os
import random
from math import log
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from stanfordcorenlp import StanfordCoreNLP

import pickle
from build_sequential_relationship import build_sequential_relationship
from build_syntactic_relationship import build_syntactic_relationship
from build_semantic_relationship import build_semantic_relationship

if len(sys.argv) < 2:
    sys.exit("Use: python build_graph.py <dataset>")

datasets = ['mr', 'ohsumed', 'R8', 'R52', 'test','liar','liar_auge']
dataset = sys.argv[1]
if dataset not in datasets:
    sys.exit("wrong dataset name")

# 滑动窗口的大小，可以自己设定，也可以自己选择
window_size = 3

# 使用没有初始过权重的图来构建图
try:
    weighted_graph = bool(sys.argv[3])
except:
    weighted_graph = False
    # print('using default unweighted graph')

print('Loading Pre-trained word vector')  # 获取预训练的词向量
print()
# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('/home/featurize/data/glove_300d/glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r',encoding='utf-8') as f:
#with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r', encoding='utf-8') as f:  # 常用英文单词的词向量
    for line in f.readlines():
        data = line.split()
        # print("The next content is about glove.6B.300d.txt:")
        # print("The data is :", data)
        # print("The data0 is :", str(data[0]))
        word_embeddings[str(data[0])] = list(map(float, data[1:]))
        # word_embedding 存储的是预训练好的单词的向量表示
        # 例如he,其表示的形式为word_embeddings[he]=[0.98239,89301929,,,,],其一共有word_embeddings_dim的维数
        # data[0]相当于表示的单词he,data[1:]相当于已经训练好的单词的向量表示

nlp = StanfordCoreNLP(r'/home/featurize/data/stanford-corenlp-full-2016-10-31', lang='en')  # 记载语法依存的工具
#nlp = StanfordCoreNLP(r'.\stanford-corenlp-full-2016-10-31', lang='en')  # 记载语法依存的工具


print('Loading document index')  # 获取跟目录下的索引文件
print()

doc_name_list = []  # 存的是test文件中的每一行，格式为 0   train   1
doc_train_list = []  # 存的是test文件中的训练集的每一行，格式为 0   train   1
doc_valid_list = []  # 存的是test文件中的验证集的每一行，格式为 0   train   1
doc_test_list = []  # 存的是test文件中的测试集的每一行，格式为 4   test   1

with open('data/' + dataset + '.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())  # line.strip 的形式为：0   train  1,与目录中的每一行内容相同
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())  # 添加到测试集中
        elif temp[1].find('valid') != -1:
            doc_valid_list.append(line.strip())  # 添加到训练集中
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())  # 添加到训练集中

print('Loading document content')  # 获取corpus目录下的新闻具体内容
print()

doc_content_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())

# map and shuffle
train_ids = []  # 获取训练数据的具体id
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)
# print("The length of train size is",len(train_ids))


valid_ids = []  # 获取验证数据的具体id
for valid_name in doc_valid_list:
    valid_id = doc_name_list.index(valid_name)
    valid_ids.append(valid_id)
random.shuffle(valid_ids)
# print("The length of train size is",len(train_ids))

test_ids = []  # 获取测试数据的具体id
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)
# print("The length of test size is",len(test_ids))

ids = train_ids + valid_ids + test_ids
# print("The train id is:", train_ids)#指的是训练数据集所在的行的行号，是打乱顺序的
# print("The test id is:", test_ids)#指的是测试数据集所在的行的行号，是打乱顺序的

# 针对的是所有的内容,打乱顺序之后的
shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    # print("The id in ids is", i)
    shuffle_doc_name_list.append(doc_name_list[int(i)])  # 编号
    shuffle_doc_words_list.append(doc_content_list[int(i)])  # 具体内容

# print("the shuffle_doc_name_list is", shuffle_doc_name_list)#获取到的是test文件中每一行的内容，如 0    train    1
# print("the shuffle_doc_words_list", shuffle_doc_words_list)#获取到的是corpus目录下的具体内容

# build corpus vocabulary
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)
# print("The words_set is:", word_set)#word_set指的是获取到的语料库中的所有单词
vocab = list(word_set)  # word_set转换为单词列表
vocab_size = len(vocab)

word_id_map = {}  # 将单词映射到数
id_word_map = {}  # 将数映射到单词
for i in range(vocab_size):
    word_id_map[vocab[i]] = i  # 将vocab中的单词映射为map图中的标号，一一对应
    id_word_map[i] = vocab[i]

# initialize out-of-vocabulary word embeddings，初始化词汇表外的单词嵌入
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)
    # 他这个意思是我先为vocab中的每一个单词都随机初始化一个向量表示，在以后真正用到该词的时候，
    # 如果我能在预训练好的字典里面找到该词，那么就用预训练好的，如果找不到的话，那就随机初始化一个向量，即用oov这个词组表示的向量来进行一个表示

# build label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)  # 获得了标签列表，仅有0，1
# print("The label list is", label_list)


# select 90% training set
train_size = len(train_ids)
# val_size = int(0.1 * train_size)  # 实际的，针对原始的数据集的
# # val_size = int(0.5 * train_size)  # 针对test数据集的
# real_train_size = train_size - val_size
valid_size = len(valid_ids)
test_size = len(test_ids)


def build_graph(start, end):  # 构建顺序矩阵
    print("0")
    x_adj = []  # 顺序矩阵
    for i in range(start, end):  #
        doc_nodes, row, col, weight = build_sequential_relationship(shuffle_doc_words_list[i], window_size)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))  # 构建了一个邻接矩阵，基于一个图中的
        x_adj.append(adj)

    return x_adj


def build_graph1(start, end):  # 构建语法矩阵
    print("1")
    x_adj1 = build_syntactic_relationship(nlp, shuffle_doc_words_list, start, end)
    return x_adj1


def build_graph2(start, end):  # 构建语义矩阵
    print("2")
    x_adj2 = []  # 语法矩阵
    semantic_pair, max_count, min_count = build_semantic_relationship(shuffle_doc_words_list, shuffle_doc_name_list,
                                                                      start,
                                                                      end)  # 先为整个数据集（就是start-end这一部分的数据），构建一个整体的语义信息，方便后面为每一句话构建语义信息来服务

    for i in range(start, end):  # tqdm是进度条的作用，为每一个文本，即每一条新闻，建立一张图
        doc_words = shuffle_doc_words_list[i].split()
        doc_nodes = len(doc_words)
        doc_word_id_map = {}  # 仅是一句话的
        for j in range(doc_nodes):
            doc_word_id_map[doc_words[j]] = j

        row = []  # 行
        col = []  # 列
        weight = []  # 语义权重
        for word1 in doc_words:
            for word2 in doc_words:
                if word1 == word2:
                    continue
                word_pair_str = str(word1) + ';' + str(word2)
                if word_pair_str in semantic_pair:
                    row.append(doc_word_id_map[word1])  # 基于局部的数字表示
                    col.append(doc_word_id_map[word2])
                    temp = (semantic_pair[word_pair_str] - min_count) / (max_count - min_count)
                    weight.append(temp)
        adj2 = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))  # 构建了一个邻接矩阵，基于一个图中的
        x_adj2.append(adj2)

    return x_adj2


def build_graph3(start, end):  # 构建特征
    print("3")
    x_feature = []

    for i in range(start, end):  # tqdm是进度条的作用，为每一个文本，即每一条新闻，建立一张图
        doc_words = shuffle_doc_words_list[i].split()
        features = []

        #方式一:自己想的
        for k in doc_words:
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])  # 将获取到的节点表示为预训练的向量

        #方式二：原始的
        # doc_vocab = list(set(doc_words))
        # doc_nodes = len(doc_vocab)
        # doc_word_id_map = {}
        # for j in range(doc_nodes):
        #     doc_word_id_map[doc_vocab[j]] = j
        # for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
        #     features.append(word_embeddings[k] if k in word_embeddings else oov[k])




        x_feature.append(features)



    return x_feature


def build_graph4(start, end):  # 构建y
    print("4")
    y = []
    # one-hot labels
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = temp[2]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)

    return y


def build_graph5(start, end):  # 统计长度
    print("5")
    doc_len_list = []  # 基于训练集或者测试集来构建的每一个句子的长度

    for i in range(start, end):  # tqdm是进度条的作用，为每一个文本，即每一条新闻，建立一张图
        doc_words = shuffle_doc_words_list[i].split()
        doc_nodes = len(doc_words)
        doc_len_list.append(doc_nodes)

    return doc_len_list


# 这里接着修改，不要一次把所有的数据都加载进来，训练集、都分开

print('building graphs for training:')
x_adj = build_graph(start=0, end=train_size)
with open("data/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)
x_adj1 = build_graph1(start=0, end=train_size)
with open("data/ind.{}.x_adj1".format(dataset), 'wb') as f:
    pkl.dump(x_adj1, f)
x_adj2 = build_graph2(start=0, end=train_size)
with open("data/ind.{}.x_adj2".format(dataset), 'wb') as f:
    pkl.dump(x_adj2, f)
x_feature = build_graph3(start=0, end=train_size)
with open("data/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)
y = build_graph4(start=0, end=train_size)
with open("data/ind.{}.x_y".format(dataset), 'wb') as f:
    pkl.dump(y, f)
doc_len_list = build_graph5(start=0, end=train_size)
print('sum_doc:', len(doc_len_list), '\n',
      'max_doc_length:', max(doc_len_list), '\n',
      'min_doc_length:', min(doc_len_list), '\n',
      'average_doc_length: {:.2f}'.format(np.mean(doc_len_list)), '\n',
      )
print()



print('building graphs for validation:')  # 验证集构建图

x_adj = build_graph(start=train_size, end=train_size+valid_size)
with open("data/ind.{}.vx_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)
x_adj1 = build_graph1(start=train_size, end=train_size+valid_size)
with open("data/ind.{}.vx_adj1".format(dataset), 'wb') as f:
    pkl.dump(x_adj1, f)
x_adj2 = build_graph2(start=train_size, end=train_size+valid_size)
with open("data/ind.{}.vx_adj2".format(dataset), 'wb') as f:
    pkl.dump(x_adj2, f)
x_feature = build_graph3(start=train_size, end=train_size+valid_size)
with open("data/ind.{}.vx_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)
y = build_graph4(start=train_size, end=train_size+valid_size)
with open("data/ind.{}.vx_y".format(dataset), 'wb') as f:
    pkl.dump(y, f)
doc_len_list = build_graph5(start=train_size, end=train_size+valid_size)
print('sum_doc:', len(doc_len_list), '\n',
      'max_doc_length:', max(doc_len_list), '\n',
      'min_doc_length:', min(doc_len_list), '\n',
      'average_doc_length: {:.2f}'.format(np.mean(doc_len_list)), '\n',
      )
print()

print('building graphs for test:')  # 最后为测试集构建图

x_adj = build_graph(start=train_size+valid_size, end=train_size+valid_size + test_size)
with open("data/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)
x_adj1 = build_graph1(start=train_size+valid_size, end=train_size+valid_size + test_size)
with open("data/ind.{}.tx_adj1".format(dataset), 'wb') as f:
    pkl.dump(x_adj1, f)
x_adj2 = build_graph2(start=train_size+valid_size, end=train_size+valid_size + test_size)
with open("data/ind.{}.tx_adj2".format(dataset), 'wb') as f:
    pkl.dump(x_adj2, f)
x_feature = build_graph3(start=train_size+valid_size, end=train_size+valid_size + test_size)
with open("data/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)
y = build_graph4(start=train_size+valid_size, end=train_size+valid_size + test_size)
with open("data/ind.{}.tx_y".format(dataset), 'wb') as f:
    pkl.dump(y, f)
doc_len_list = build_graph5(start=train_size+valid_size, end=train_size+valid_size + test_size)
print('sum_doc:', len(doc_len_list), '\n',
      'max_doc_length:', max(doc_len_list), '\n',
      'min_doc_length:', min(doc_len_list), '\n',
      'average_doc_length: {:.2f}'.format(np.mean(doc_len_list)), '\n',
      )
print()
