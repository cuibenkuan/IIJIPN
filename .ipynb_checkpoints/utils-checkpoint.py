# 工具函数的定义

import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import re
from tqdm import tqdm


# import sparse


def parse_index_file(filename):  # 处理index文件并返回index矩阵
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):  # 创建 mask 并返回mask矩阵
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # 读取数据
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_adj1', 'x_adj2', 'x_embed', 'x_y', 'tx_adj', 'tx_adj1', 'tx_adj2', 'tx_embed', 'tx_y',
             'vx_adj', 'vx_adj1', 'vx_adj2', 'vx_embed', 'vx_y']
    objects = []
    for i in range(len(names)):  # 从build_graph文件中生成的9个文件中读取数据
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):  # 判断python是否为3.0以上（包括3.0）
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_adj1, x_adj2, x_embed, y, tx_adj, tx_adj1, tx_adj2, tx_embed, ty, vx_adj, vx_adj1, vx_adj2, vx_embed, vy = tuple(
        objects)  # tuple() 函数将列表转换为元组,得到的返回值是括号括起来的，形式为（1，2，3）
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)
    # x_adj : <class 'list'>

    train_adj = []
    train_adj1 = []
    train_adj2 = []
    train_embed = []

    val_adj = []
    val_adj1 = []
    val_adj2 = []
    val_embed = []

    test_adj = []
    test_adj1 = []
    test_adj2 = []
    test_embed = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        adj1 = x_adj1[i].toarray()
        adj2 = x_adj2[i].toarray()


        train_adj.append(adj)
        train_adj1.append(adj1)
        train_adj2.append(adj2)

        embed = np.array(x_embed[i])
        train_embed.append(embed)

    for i in range(len(vy)):  # valid_size:

        adj = vx_adj[i].toarray()
        adj1 = vx_adj1[i].toarray()
        adj2 = vx_adj2[i].toarray()

        val_adj.append(adj)
        val_adj1.append(adj1)
        val_adj2.append(adj2)

        embed = np.array(vx_embed[i])
        val_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        adj1 = tx_adj1[i].toarray()
        adj2 = tx_adj2[i].toarray()

        test_adj.append(adj)
        test_adj1.append(adj1)
        test_adj2.append(adj2)

        embed = np.array(tx_embed[i])
        test_embed.append(embed)

    train_adj = np.array(train_adj)  # <class 'numpy.ndarray'>
    train_adj1 = np.array(train_adj1)
    train_adj2 = np.array(train_adj2)

    val_adj = np.array(val_adj)
    val_adj1 = np.array(val_adj1)
    val_adj2 = np.array(val_adj2)

    test_adj = np.array(test_adj)
    test_adj1 = np.array(test_adj1)
    test_adj2 = np.array(test_adj2)

    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)

    train_y = np.array(y)
    val_y = np.array(vy)  # train_size])
    test_y = np.array(ty)

    # 返回的其实是有向图的矩阵，即对于adj,adj1,adj2来说，他们都是非对称矩阵
    # 在TensoGCN中，是将非对称邻接矩阵adj,adj1,adj2转变为对称邻接矩阵(有向图转无向图)，即
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    # adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    # 在我们的代码中，是否要将其转换为对称矩阵（无向图来看待），看后面的需求

    return train_adj, train_adj1, train_adj2, train_embed, train_y, val_adj, val_adj1, val_adj2, val_embed, val_y, test_adj, test_adj1, test_adj2, test_embed, test_y


# def sparse_to_tuple(sparse_mx):  # 将矩阵转换成tuple格式并返回
#     """Convert sparse matrix to tuple representation."""
#
#     def to_tuple(mx):
#         if not sp.isspmatrix_coo(mx):
#             mx = mx.tocoo()
#         coords = np.vstack((mx.row, mx.col)).transpose()
#         values = mx.data
#         shape = mx.shape
#         return coords, values, shape
#
#     if isinstance(sparse_mx, list):
#         for i in range(len(sparse_mx)):
#             sparse_mx[i] = to_tuple(sparse_mx[i])
#     else:
#         sparse_mx = to_tuple(sparse_mx)
#
#     return sparse_mx


def normalize_adj(adj):  # 图归一化并返回，归一化：把矩阵中的每一个数据映射到0～1范围之内，使得之后的处理更加便捷快速
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):  # 处理得到GCN中的归一化矩阵并返回
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1))  # mask for padding

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i])  # #将原来的矩阵归一化，矩阵中的每一个元素都转化为0-1直接的数
        pad = max_length - adj_normalized.shape[0]  # 目的就是填充，使得每一个句子的矩阵都具有相同的形状，因为句子的长短不一，也就造成初始的每个矩阵形状不统一，这个目的就是
        # 使得每个句子的矩阵都具有相同的形状
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')  # 补出来的和最大的矩阵的形状一致，不够的行或者列用0来填充
        mask[i, :adj[i].shape[0], :] = 1.  # mask的作用是标识填充完之后的矩阵，是n×1的矩阵，原始的行用1表示，补充的用0表示
        adj[i] = adj_normalized
    # print(type(adj))#<class 'numpy.ndarray'>
    # print(type(np.array(list(adj))))#<class 'numpy.ndarray'>
    # return sparse_to_tuple(adj), mask
    return np.array(list(adj)), mask


def preprocess_features(features):  # 处理特征:将特征进行归一化并返回tuple (coords, values, shape)
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = max([len(f) for f in features])

    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    return np.array(list(features))


def construct_feed_dict(features, adj, adj1, adj2, mask, mask1, mask2, labels, placeholders):  # 构建输入字典并返回
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})

    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['adj1']: adj1})
    feed_dict.update({placeholders['adj2']: adj2})

    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['mask1']: mask1})
    feed_dict.update({placeholders['mask2']: mask2})

    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

