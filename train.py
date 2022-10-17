

from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from utils import *
from models import GNN
import os

if len(sys.argv) != 2:
    sys.exit("Use: python train.py <dataset>")

datasets = ['constraint']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
flags.DEFINE_string('model', 'gnn', 'Model string.')
flags.DEFINE_float('learning_rate',  0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 128, 'Size of batches per epoch.')
flags.DEFINE_integer('input_dim', 300, 'Dimension of input.')
flags.DEFINE_integer('hidden', 64, 'Number of units in hidden layer.')
flags.DEFINE_integer('steps', 1, 'Number of graph layers.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.001, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping',20,'Tolerance for early stopping (# of epochs).')

# Load data
train_adj, train_adj1, train_adj2, train_feature, train_y, val_adj, val_adj1, val_adj2, val_feature, val_y, test_adj, test_adj1, test_adj2, test_feature, test_y = load_data(
    FLAGS.dataset)


print('loading training set')
train_adj, train_mask = preprocess_adj(train_adj)
train_adj1, train_mask1 = preprocess_adj(train_adj1)
train_adj2, train_mask2 = preprocess_adj(train_adj2)
train_feature = preprocess_features(train_feature)
print('loading validation set')
val_adj, val_mask = preprocess_adj(val_adj)
val_adj1, val_mask1 = preprocess_adj(val_adj1)
val_adj2, val_mask2 = preprocess_adj(val_adj2)
val_feature = preprocess_features(val_feature)
print('loading test set')
test_adj, test_mask = preprocess_adj(test_adj)
test_adj1, test_mask1 = preprocess_adj(test_adj1)
test_adj2, test_mask2 = preprocess_adj(test_adj2)
test_feature = preprocess_features(test_feature)

placeholders = {
    'adj': tf.placeholder(tf.float32, shape=(None, None, None)),
    'adj1': tf.placeholder(tf.float32, shape=(None, None, None)),
    'adj2': tf.placeholder(tf.float32, shape=(None, None, None)),

    'mask': tf.placeholder(tf.float32, shape=(None, None, 1)),
    'mask1': tf.placeholder(tf.float32, shape=(None, None, 1)),
    'mask2': tf.placeholder(tf.float32, shape=(None, None, 1)),

    'features': tf.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
    'labels': tf.placeholder(tf.float32, shape=(None, train_y.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)
}


model = GNN(placeholders, input_dim=FLAGS.input_dim, logging=True)

sess = tf.Session()



def evaluate(features, adj, adj1, adj2, mask, mask1, mask2, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, adj, adj1, adj2, mask, mask1, mask2, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]



sess.run(tf.global_variables_initializer())

cost_val = []
best_val = 0
best_epoch = 0
best_acc = 0
best_cost = 0
best_test = 0
test_doc_embeddings = None
preds = None
labels = None



print('train start...')
# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()

    # Training step
    indices = np.arange(0, len(train_y))
    np.random.shuffle(indices)

    train_loss, train_acc = 0, 0
    for start in range(0, len(train_y), FLAGS.batch_size):
        end = start + FLAGS.batch_size
        idx = indices[start:end]
        # Construct feed dictionary
        feed_dict = construct_feed_dict(train_feature[idx], train_adj[idx], train_adj1[idx], train_adj2[idx],
                                        train_mask[idx], train_mask[idx], train_mask[idx],
                                        train_y[idx], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        train_loss += outs[1] * len(idx)
        train_acc += outs[2] * len(idx)
    train_loss /= len(train_y)
    train_acc /= len(train_y)

    # Validation
    val_cost, val_acc, val_duration, _, _, _ = evaluate(val_feature, val_adj, val_adj1, val_adj2, val_mask, val_mask,
                                                        val_mask, val_y,
                                                        placeholders)
    cost_val.append(val_cost)

    # Test
    test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(test_feature, test_adj, test_adj1,
                                                                            test_adj2,
                                                                            test_mask, test_mask, test_mask, test_y,
                                                                            placeholders)

    if val_acc >= best_val:
        best_val = val_acc
        best_epoch = epoch
        best_acc = test_acc
        best_cost = test_cost
        test_doc_embeddings = embeddings
        preds = pred

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
              "val_acc=", "{:.5f}".format(val_acc),"test_loss=", "{:.5f}".format(test_cost), "test_acc=", "{:.5f}".format(test_acc),
              "time=", "{:.5f}".format(time.time() - t),"**")

    else:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
              "val_acc=", "{:.5f}".format(val_acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=", "{:.5f}".format(test_acc),
              "time=", "{:.5f}".format(time.time() - t))



    if FLAGS.early_stopping > 0 and epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
            cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")
# Best results
print('Best epoch:', best_epoch)
print("Test set results:", "cost=", "{:.5f}".format(best_cost),
      "accuracy=", "{:.5f}".format(best_acc))

