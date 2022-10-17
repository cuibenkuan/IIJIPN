import numpy as np
import pickle as pkl
import sys

from tqdm import tqdm



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
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
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_adj1, x_adj2, x_embed, y, tx_adj, tx_adj1, tx_adj2, tx_embed, ty, vx_adj, vx_adj1, vx_adj2, vx_embed, vy = tuple(
        objects)

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

    train_adj = np.array(train_adj)
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
    val_y = np.array(vy)
    test_y = np.array(ty)

    return train_adj, train_adj1, train_adj2, train_embed, train_y, val_adj, val_adj1, val_adj2, val_embed, val_y, test_adj, test_adj1, test_adj2, test_embed, test_y




def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1))  # mask for padding

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i])
        pad = max_length - adj_normalized.shape[0]
        adj_normalized = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adj[i].shape[0], :] = 1.
        adj[i] = adj_normalized
    return np.array(list(adj)), mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = max([len(f) for f in features])

    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    return np.array(list(features))


def construct_feed_dict(features, adj, adj1, adj2, mask, mask1, mask2, labels, placeholders):
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

