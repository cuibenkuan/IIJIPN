#评测指标的计算




import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def softmax_cross_entropy(preds, labels):#交叉损失函数
    """Softmax cross-entropy loss with masking."""
    print("softmax_cross_entropy")
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):#预测值
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)
