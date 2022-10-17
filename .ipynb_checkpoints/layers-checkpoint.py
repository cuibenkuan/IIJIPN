# GCN层的定义


from inits import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def sparse_dense_matmul_batch(sp_a, b):  # Tensor中没有

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)


def dot(x, y, sparse=False):  # 和Tensor中dot2函数相同
    """Wrapper for 3D tf.matmul (sparse vs dense)."""
    if sparse:
        res = sparse_dense_matmul_batch(x, y)  # 向量乘法，和稀疏矩阵有关
    else:
        res = tf.einsum('bij,jk->bik', x, y)  # tf.matmul(x, y)
    return res


def intra_propagations(i, support, x, var, act, mask, dropout, sparse_inputs=False):  # 图内传播,用的GRU_unit 更新单元

    #print("layers 58")

    # support = intra_propagations(i, self.adj, output, self.vars, self.act,
    #                              self.mask, 1 - self.dropout, self.sparse_inputs)  # GRU更新模块,support为矩阵，x为特征
    """GRU unit with 3D tensor inputs."""
    # message passing
    support = tf.nn.dropout(support, dropout)  # optional
    a = tf.matmul(support, x)

    # update gate        
    z0 = dot(a, var['weights_' + str(i) + '_z0'], sparse_inputs) + var['bias_' + str(i) + '_z0']
    z1 = dot(x, var['weights_' + str(i) + '_z1'], sparse_inputs) + var['bias_' + str(i) + '_z1']
    z = tf.sigmoid(z0 + z1)

    # reset gate
    r0 = dot(a, var['weights_' + str(i) + '_r0'], sparse_inputs) + var['bias_' + str(i) + '_r0']
    r1 = dot(x, var['weights_' + str(i) + '_r1'], sparse_inputs) + var['bias_' + str(i) + '_r1']
    r = tf.sigmoid(r0 + r1)

    # update embeddings    
    h0 = dot(a, var['weights_' + str(i) + '_h0'], sparse_inputs) + var['bias_' + str(i) + '_h0']
    h1 = dot(r * x, var['weights_' + str(i) + '_h1'], sparse_inputs) + var['bias_' + str(i) + '_h1']
    h = act(mask * (h0 + h1))

    return h * z + x * (1 - z)


def inter_propagation(var, support, support1, support2):  # 图间的传播

    i = 0
    support = dot(support, var['weights_' + str(i) + str(i)]) + var['bias_' + str(i) + str(i)]
    i = 1
    support1 = dot(support1, var['weights_' + str(i) + str(i)]) + var['bias_' + str(i) + str(i)]
    i = 2
    support2 = dot(support2, var['weights_' + str(i) + str(i)]) + var['bias_' + str(i) + str(i)]

    att_features = []
    #方式一：相加
    att_features.append(tf.nn.leaky_relu(tf.add(support, support1)))
    att_features.append(tf.nn.leaky_relu(tf.add(support1, support2)))
    att_features.append(tf.nn.leaky_relu(tf.add(support2, support)))
    #方式二：直接返回
    # att_features.append(support)
    # att_features.append(support1)
    # att_features.append(support2)


    return att_features


def attentin_graps(supports, var):  # 基于图的注意力机制
    #print("104")

    i = 0

    supports[i] = dot(supports[i], var['weights_att' + str(i)]) + var['bias_att' + str(i)]

    i = 1

    supports[i] = dot(supports[i], var['weights_att' + str(i)]) + var['bias_att' + str(i)]

    i = 2

    supports[i] = dot(supports[i], var['weights_att' + str(i)]) + var['bias_att' + str(i)]


    #实现方式1：比例相加
    #print("125") tf.add() tf.subtract() tf.multiply() tf.divide()  加减乘除
    # sum = tf.add(supports[0] , supports[1])
    # sum = tf.add(sum, supports[2])
    #
    # mul = tf.multiply(supports[0],supports[0])
    # mul1 = tf.multiply(supports[1], supports[1])
    # mul2 = tf.multiply(supports[2], supports[2])
    #
    # div = tf.div_no_nan(mul,sum)
    # div1 = tf.div_no_nan(mul1, sum)
    # div2 = tf.div_no_nan(mul2, sum)
    #
    #
    # end = tf.add(div,div1)
    # end = tf.add(end,div2)
    #
    # return end

    #实现方式2：直接相加
    supports[0] = tf.add(supports[0], supports[1])
    supports[0] = tf.add(supports[0], supports[2])

    return supports[0]




class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()，封装_call()函数
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

        #print("layers 145")

    def _call(self, inputs):
        #print("layers,148")
        return inputs

    def __call__(self, inputs):  # 在model里面，对层进行具体的运算操作时，是先进入到这儿的
        #print("layers,151")
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            #print("layers 156")
            outputs = self._call(inputs)  # 这儿，每一层就分别调用各自的_call函数了
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphLayer(Layer):  # GNN中有用到，也是就是代码中用到的模块
    """Graph layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, steps=2, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act

        self.adj = placeholders['adj']
        self.adj1 = placeholders['adj1']
        self.adj2 = placeholders['adj2']

        self.mask = placeholders['mask']
        self.mask1 = placeholders['mask1']
        self.mask2 = placeholders['mask2']

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.steps = steps

        # self.input_dim = input_dim  # Tensor中有的
        # self.output_dim = output_dim  # Tensor中有的

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        #print("layers 197")

        with tf.variable_scope(self.name + '_vars'):

            # tf.get_variable()是什么意思？
            # 例子: tf.get_variable("v")
            # 获取v的变量，如果变量名为v的变量没有就创建，有的话就引用；
            # assert又是什么意思？
            # 是断言，判断真假，如果为真就不报错继续执行，如果为假就抛出异常；
            # 现在我们来理解，tf.variable_scope()
            # 的意义，在大项目中，变量很多，第114行有一个v的变量，第339行又出现了v的变量，后面写代码都晕了，搞不清楚。怎么办？搞一个变量管理器，即使变量名一样，但是变量作用域不一样，引用的时候就不会出现穿插问题了，方便代码的维护；

            # self.vars['weights_encode'] = glorot([input_dim, output_dim],name='weights_encode')
            # self.vars['weights_z0'] = glorot([output_dim, output_dim], name='weights_z0')#glorot为一种权重初始化的方式，在inits中有重新定义这个函数
            # self.vars['weights_z1'] = glorot([output_dim, output_dim], name='weights_z1')
            # self.vars['weights_r0'] = glorot([output_dim, output_dim], name='weights_r0')
            # self.vars['weights_r1'] = glorot([output_dim, output_dim], name='weights_r1')
            # self.vars['weights_h0'] = glorot([output_dim, output_dim], name='weights_h0')
            # self.vars['weights_h1'] = glorot([output_dim, output_dim], name='weights_h1')
            #
            # self.vars['bias_encode'] = zeros([output_dim], name='bias_encode')#在inits中有重新定义zeros这个函数
            # self.vars['bias_z0'] = zeros([output_dim], name='bias_z0')
            # self.vars['bias_z1'] = zeros([output_dim], name='bias_z1')
            # self.vars['bias_r0'] = zeros([output_dim], name='bias_r0')
            # self.vars['bias_r1'] = zeros([output_dim], name='bias_r1')
            # self.vars['bias_h0'] = zeros([output_dim], name='bias_h0')
            # self.vars['bias_h1'] = zeros([output_dim], name='bias_h1')

            #print("layers 224")

            for i in range(3):
                self.vars['weights_encode_' + str(i)] = glorot([input_dim, output_dim],
                                                               name='weights_encode_' + str(i))  # 对输入的X先进行一次编码
                self.vars['bias_encode_' + str(i)] = zeros([output_dim], name='bias_encode_' + str(i))

                self.vars['weights_' + str(i) + str(i)] = glorot([output_dim, output_dim],
                                                                 name='weights_' + str(i) + str(i))  # 图间传播更新
                self.vars['bias_' + str(i) + str(i)] = zeros([output_dim], name='bias_' + str(i) + str(i))

                self.vars['weights_' + str(i) + '_z0'] = glorot([output_dim, output_dim],
                                                                name='weights_' + str(i) + '_z0')  # 图内传播更新,w
                self.vars['weights_' + str(i) + '_z1'] = glorot([output_dim, output_dim],
                                                                name='weights_' + str(i) + '_z1')
                self.vars['weights_' + str(i) + '_r0'] = glorot([output_dim, output_dim],
                                                                name='weights_' + str(i) + '_r0')
                self.vars['weights_' + str(i) + '_r1'] = glorot([output_dim, output_dim],
                                                                name='weights_' + str(i) + '_r1')
                self.vars['weights_' + str(i) + '_h0'] = glorot([output_dim, output_dim],
                                                                name='weights_' + str(i) + '_h0')
                self.vars['weights_' + str(i) + '_h1'] = glorot([output_dim, output_dim],
                                                                name='weights_' + str(i) + '_h1')

                self.vars['bias_' + str(i) + '_z0'] = zeros([output_dim], name='bias_' + str(i) + '_z0')  # 图内传播更新,b
                self.vars['bias_' + str(i) + '_z1'] = zeros([output_dim], name='bias_' + str(i) + '_z1')
                self.vars['bias_' + str(i) + '_r0'] = zeros([output_dim], name='bias_' + str(i) + '_r0')
                self.vars['bias_' + str(i) + '_r1'] = zeros([output_dim], name='bias_' + str(i) + '_r1')
                self.vars['bias_' + str(i) + '_h0'] = zeros([output_dim], name='bias_' + str(i) + '_h0')
                self.vars['bias_' + str(i) + '_h1'] = zeros([output_dim], name='bias_' + str(i) + '_h1')

                self.vars['weights_att' + str(i)] = glorot([output_dim, output_dim],
                                                           name='weights_att' + str(i))  # 图之间的attention机制
                self.vars['bias_att' + str(i)] = zeros([output_dim], name='bias_att' + str(i))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        '''
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # encode inputs
        x = dot(x, self.vars['weights_encode'], #稀疏输入模块
                self.sparse_inputs) + self.vars['bias_encode']
        output = self.mask * self.act(x)

        # convolve
        for _ in range(self.steps):#图内更新
            output = gru_unit(self.support, output, self.vars, self.act,
                              self.mask, 1-self.dropout, self.sparse_inputs)#GRU更新模块

        return output
        '''

        #print("layers 282")
        xx = inputs

        # dropout
        if self.sparse_inputs:
            xx = sparse_dropout(xx, 1 - self.dropout, self.num_features_nonzero)
        else:
            xx = tf.nn.dropout(xx, 1 - self.dropout)  # nn.dropout是将输入中的部分元素设置为0.对于每次前向调用，被置0的元素都是随机的

        # convolve

        #print("layers 301")

        # 图内传播，三个图分别进行图内传播
        i = 0  # 顺序关系
        x = dot(xx, self.vars['weights_encode_' + str(i)], self.sparse_inputs) + self.vars['bias_encode_' + str(i)]
        #print("layers 306")
        support = self.mask * self.act(x)
        #print("layers 308")
        for _ in range(self.steps):
            support = intra_propagations(i, self.adj, support, self.vars, self.act,
                                         self.mask, 1 - self.dropout, self.sparse_inputs)  # GRU更新模块

        #print("layers 302")

        i = 1  # 语义关系
        x = dot(xx, self.vars['weights_encode_' + str(i)],  # 稀疏输入模块,图内传播
                self.sparse_inputs) + self.vars['bias_encode_' + str(i)]
        support1 = self.mask1 * self.act(x)
        for _ in range(self.steps):
            support1 = intra_propagations(i, self.adj1, support1, self.vars, self.act,
                                          self.mask1, 1 - self.dropout, self.sparse_inputs)  # GRU更新模块

        #print("layers 303")

        i = 2  # 语法关系
        x = dot(xx, self.vars['weights_encode_' + str(i)],  # 稀疏输入模块,图内传播
                self.sparse_inputs) + self.vars['bias_encode_' + str(i)]
        support2 = self.mask2 * self.act(x)
        for _ in range(self.steps):
            support2 = intra_propagations(i, self.adj2, support2, self.vars, self.act,
                                          self.mask2, 1 - self.dropout, self.sparse_inputs)  # GRU更新模块

        #print("layers 304")

        supports = inter_propagation(self.vars, support, support1, support2)  # 调用上面的函数,进行图间的传播

        #print("layers 306")

        supports = attentin_graps(supports, self.vars)#调用上面的函数，进行图的attention机制，来

        #print("layers 308")

        return supports


class ReadoutLayer(Layer):  # GNN中有用到，也是就是代码中用到的模块
    """Graph Readout Layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.mask = placeholders['mask']

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_att'] = glorot([input_dim, 1], name='weights_att')
            self.vars['weights_emb'] = glorot([input_dim, input_dim], name='weights_emb')
            self.vars['weights_mlp'] = glorot([input_dim, output_dim], name='weights_mlp')

            self.vars['bias_att'] = zeros([1], name='bias_att')
            self.vars['bias_emb'] = zeros([input_dim], name='bias_emb')
            self.vars['bias_mlp'] = zeros([output_dim], name='bias_mlp')

        if self.logging:
            self._log_vars()

        #print("layers 361")

    def _call(self, inputs):

        #print("layers 381")
        x = inputs

        # soft attention
        att = tf.sigmoid(dot(x, self.vars['weights_att']) + self.vars['bias_att'])
        emb = self.act(dot(x, self.vars['weights_emb']) + self.vars['bias_emb'])

        N = tf.reduce_sum(self.mask, axis=1)
        M = (self.mask - 1) * 1e9

        # graph summation
        g = self.mask * att * emb
        g = tf.reduce_sum(g, axis=1) / N + tf.reduce_max(g + M, axis=1)
        g = tf.nn.dropout(g, 1 - self.dropout)

        # classification
        output = tf.matmul(g, self.vars['weights_mlp']) + self.vars['bias_mlp']
        #print("layers 388")

        return output
