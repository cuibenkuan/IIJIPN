# 模型结构定义


from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.embeddings = None#TextING特有的

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        # print("models 39")

    def _build(self):
        # print("build 42")
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        # print("build 46")

        # Build sequential layer model  TextING的

        # print("The self.inputs")
        # print(self.inputs)
        self.activations = [self.inputs]  # self.inputs为features

        # print("The self.activations")
        # print(self.activations)

        # print("The self.activations[-1]")
        # print(self.activations[-1])#索引-1代表最后一个元素

        for layer in self.layers:  # self.layers包含两层模型，是由下面的函数添加的，包含一层graphlayer,一层readoutlayer,当时仅进行了初始化进行了声明，并没有进行层的具体运算操作
            # print("models 52")
            hidden = layer(self.activations[-1])  # 这儿，开始对每一层进行具体的操作，前面仅仅是声明初始化了一个层，并没有进行具体的运算
            # print("models 54")
            self.activations.append(hidden)

        self.embeddings = self.activations[-2]
        self.outputs = self.activations[-1]

        '''
        
        self.activations.append(self.inputs)#TensorGCN中的
        self.activations.append(self.inputs)
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-3:])
            self.activations.extend(hidden)
        self.outputs = tf.stack([self.activations[-3], self.activations[-2], self.activations[-1]], axis=0)
        self.outputs = tf.reduce_mean(self.outputs, axis=0)
        
        '''
        #print("models,86")
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        #print("models,87")
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GNN(Model):  # 有用到
    def __init__(self, placeholders, input_dim, **kwargs):
        # 如果我们不确定要往函数中传入多少个参数，或者我们想往函数中以列表和元组的形式传参数时，那就使要用 * args；
        # 如果我们不知道要往函数中传入多少个关键词参数，或者想传入字典的值作为关键词参数时，那就要使用 ** kwargs。
        super(GNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.mask = placeholders['mask']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        print('build...')
        self.build()  # 接着进行下面的146行的_build函数

    def _loss(self):
        #print("models,141")
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if 'weights' in var.name or 'bias' in var.name:
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        #print("models,154")
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

        self.preds = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):  # 在127行调用完这个函数之后，对两个layer进行了初始化的操作，并没有进行具体的运算操作，在调用完这个函数之后，又调用了上面Model模块中的build函数，即第45行的函数
        #print("model 146")
        # self.layers是GNN模型中的一个元素，往里面添加两个对象，分别为GraphLayer和ReadoutLayer,这儿其实只相当于对两个layer进行了初始化，只调用了各自的__init__函数，有了那么一个对象，但是还没有进行具体的数据进行操作
        self.layers.append(GraphLayer(input_dim=self.input_dim,
                                      output_dim=FLAGS.hidden,
                                      placeholders=self.placeholders,
                                      act=tf.tanh,
                                      sparse_inputs=False,
                                      dropout=True,
                                      steps=FLAGS.steps,
                                      logging=self.logging))
        #print("model 155")
        self.layers.append(ReadoutLayer(input_dim=FLAGS.hidden,
                                        output_dim=self.output_dim,
                                        placeholders=self.placeholders,
                                        act=tf.tanh,
                                        sparse_inputs=False,
                                        dropout=True,
                                        logging=self.logging))

    def predict(self):
        #print("models,180")
        return tf.nn.softmax(self.outputs)
