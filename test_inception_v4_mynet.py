import numpy as np
import tensorflow as tf
import time
import sys
import math
import gc
import dogs_inception_v4
slim = tf.contrib.slim
"""
epoch 160
('Train Accuracy:', 0.96551305)
('Test Accuracy:', 0.85036677)
epoch 800
('Train Accuracy:', 0.9952305)
('Test Accuracy:', 0.84547675)



"""

FILEXTRAIN = './cache/inception_v4/my_network_train_x.npy'
FILEYTRAIN = './cache/inception_v4/my_network_train_y.npy'
FILEXTEST = './cache/inception_v4/my_network_test_x.npy'
FILEYTEST = './cache/inception_v4/my_network_test_y.npy'
FILECOLUMNS = './cache/inception_v4/my_network_y_columns.npy'

TRAINEDMODEL = '/root/kaggle/dogs/model.ckpt'

def getDataTest():
    X_train = np.load(FILEXTRAIN)
    y_train = np.load(FILEYTRAIN)
    X_test = np.load(FILEXTEST)
    y_test = np.load(FILEYTEST)
    return X_train,y_train,X_test,y_test

def dense(x, size, scope):
	return tf.contrib.layers.fully_connected(x, size,activation_fn=None,scope=scope)

def dense_batch_relu(x, size, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, 
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                          center=True, scale=True, 
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')
  
        
def getMyNetwork(n_outputs):
    X_t = tf.placeholder(shape = [ None, 8,8,1536],dtype = tf.float32, name = "X_t")
    y_t = tf.placeholder(shape =[ None, n_outputs],dtype = tf.float32, name = "y_t")
    phase = tf.placeholder(tf.bool, name='phase')
    # 8 x 8 x 1536
    net = slim.avg_pool2d(X_t, [8,8], padding='VALID',  scope='AvgPool_1a')
    # 1 x 1 x 1536
    PreLogitsFlatten = slim.flatten(net, scope='PreLogitsFlatten')
    # 1536
    h1 = dense_batch_relu(PreLogitsFlatten, 960, phase,'layer1')
    h2 = dense_batch_relu(h1, 480, phase, 'layer2')
    h3 = dense_batch_relu(h2, 240, phase, 'layer3')
    logits = dense(h3, n_outputs, 'logits')
    return X_t, y_t, logits, phase

X_train,y_train,X_test,y_test = getDataTest()
m = y_test.shape[0]
n_y = y_test[0].shape[0]
# get network
X_t, y_t, logits, phase = getMyNetwork(n_y)

saver = tf.train.Saver()


# Ensures that we execute the update_ops before performing the train_step
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_t, 1), tf.argmax(logits, 1)),'float32'))

with tf.Session() as sess:
	# Restore variables from disk.
  	saver.restore(sess, TRAINEDMODEL)
	print ("Train Accuracy:", accuracy.eval({X_t: X_train, y_t: y_train,phase: 0}))
	print ("Test Accuracy:", accuracy.eval({X_t: X_test, y_t: y_test,phase: 0}))


