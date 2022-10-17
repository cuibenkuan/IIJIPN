import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from stanfordcorenlp import StanfordCoreNLP


from build_sequential_relationship import build_sequential_relationship
from build_syntactic_relationship import build_syntactic_relationship
from build_semantic_relationship import build_semantic_relationship

if len(sys.argv) < 2:
    sys.exit("Use: python build_graph.py <dataset>")

datasets = ['constraint']
dataset = sys.argv[1]
if dataset not in datasets:
    sys.exit("wrong dataset name")

window_size = 3

print('Loading Pre-trained word vector')


word_embeddings_dim = 300
word_embeddings = {}


with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))


nlp = StanfordCoreNLP(r'.\stanford-corenlp-full-2016-10-31', lang='en')


print('Loading document index')
print()

doc_name_list = []
doc_train_list = []
doc_valid_list = []
doc_test_list = []

with open('data/' + dataset + '.txt', 'r') as f:
    for line in f.readlines():
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('valid') != -1:
            doc_valid_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())

print('Loading document content')
print()

doc_content_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())


train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)



valid_ids = []
for valid_name in doc_valid_list:
    valid_id = doc_name_list.index(valid_name)
    valid_ids.append(valid_id)
random.shuffle(valid_ids)


test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + valid_ids + test_ids

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])

# build corpus vocabulary
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    word_set.update(words)

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
id_word_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i
    id_word_map[i] = vocab[i]


oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

# build label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)




train_size = len(train_ids)
valid_size = len(valid_ids)
test_size = len(test_ids)


def build_graph(start, end):
    x_adj = []
    for i in range(start, end):
        doc_nodes, row, col, weight = build_sequential_relationship(shuffle_doc_words_list[i], window_size)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        x_adj.append(adj)

    return x_adj


def build_graph1(start, end):
    x_adj1 = build_syntactic_relationship(nlp, shuffle_doc_words_list, start, end)
    return x_adj1


def build_graph2(start, end):
    x_adj2 = []
    semantic_pair, max_count, min_count = build_semantic_relationship(shuffle_doc_words_list, shuffle_doc_name_list,
                                                                      start,
                                                                      end)

    for i in range(start, end):
        doc_words = shuffle_doc_words_list[i].split()
        doc_nodes = len(doc_words)
        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_words[j]] = j

        row = []
        col = []
        weight = []
        for word1 in doc_words:
            for word2 in doc_words:
                if word1 == word2:
                    continue
                word_pair_str = str(word1) + ';' + str(word2)
                if word_pair_str in semantic_pair:
                    row.append(doc_word_id_map[word1])
                    col.append(doc_word_id_map[word2])
                    temp = (semantic_pair[word_pair_str] - min_count) / (max_count - min_count)
                    weight.append(temp)
        adj2 = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        x_adj2.append(adj2)

    return x_adj2


def build_graph3(start, end):

    x_feature = []

    for i in range(start, end):
        doc_words = shuffle_doc_words_list[i].split()
        features = []

        for k in doc_words:
            features.append(word_embeddings[k] if k in word_embeddings else oov[k])

        x_feature.append(features)



    return x_feature


def build_graph4(start, end):

    y = []

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


def build_graph5(start, end):

    doc_len_list = []

    for i in range(start, end):
        doc_words = shuffle_doc_words_list[i].split()
        doc_nodes = len(doc_words)
        doc_len_list.append(doc_nodes)

    return doc_len_list




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


print('building graphs for validation:')

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

print('building graphs for test:')

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
