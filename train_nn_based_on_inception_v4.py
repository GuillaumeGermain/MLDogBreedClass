import numpy as np
import tensorflow as tf
import time
import sys
import math
import os.path
import tflearn
import pandas as pd
import dogs_inception_v4
from sklearn.model_selection import train_test_split
slim = tf.contrib.slim
"""

        with tf.variable_scope('Logits'):
          # 8 x 8 x 1536
          net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                scope='AvgPool_1a')
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
          end_points['PreLogitsFlatten'] = net
          # 1536
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')


"""

FILEXTRAIN = './cache/inception_v4/my_network_train_x.npy'
FILEYTRAIN = './cache/inception_v4/my_network_train_y.npy'
FILEXTEST = './cache/inception_v4/my_network_test_x.npy'
FILEYTEST = './cache/inception_v4/my_network_test_y.npy'
FILECOLUMNS = './cache/inception_v4/my_network_y_columns.npy'

TRAINEDMODEL = '/root/kaggle/dogs/model.ckpt'

starter_learning_rate = 0.00001
# learning_rate = 0.000001
mini_batch_size = 64
epoch_control = 10
num_epochs = 5001
epoch_learning_rate_decay = 100
# L2 regularisation
# reg_constant=0.08
reg_constant=0.2
# dropout keep probability
dropout_keep_prob = 0.5

def getData():
    if (os.path.isfile(FILEXTRAIN)):
        X_train = np.load(FILEXTRAIN)
        y_train = np.load(FILEYTRAIN)
        X_test = np.load(FILEXTEST)
        y_test = np.load(FILEYTEST)
    else:
        X,y = dogs_inception_v4.get_data_from_pretrained_Inceptionv4('Mixed_7d')
        y_dum = pd.get_dummies(y)
        column_names = y_dum.columns.values.tolist()
        X_train, X_test, y_train, y_test = train_test_split( X, y_dum.as_matrix(), test_size=0.2)
        np.save(FILECOLUMNS,column_names)
        np.save(FILEXTRAIN,X_train)
        np.save(FILEYTRAIN,y_train)
        np.save(FILEXTEST,X_test)
        np.save(FILEYTEST,y_test)
    return X_train, y_train, X_test, y_test

def dense(x, size, scope):
	return tf.contrib.layers.fully_connected(x, size,activation_fn=None,scope=scope)
  
def dense_batch_relu(x, size, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size, 
                                           activation_fn=None,
                                           scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                      center=True, 
                                      scale=True, 
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
    def f2(net): return net
    net = tf.cond(phase, lambda: slim.dropout(net, dropout_keep_prob, scope='Dropout_1b'), lambda: f2(net)) 
    net = slim.flatten(net, scope='PreLogitsFlatten')
    # 1536
    net = dense_batch_relu(net, 960, phase,'layer1')
    net = tf.cond(phase, lambda: tf.nn.dropout(net, dropout_keep_prob), lambda: f2(net))
    net = dense_batch_relu(net, 480, phase, 'layer2')
    net = tf.cond(phase, lambda: tf.nn.dropout(net, dropout_keep_prob), lambda: f2(net))
    net = dense_batch_relu(net, 240, phase, 'layer3')
    logits = dense(net, n_outputs, 'logits')
    return X_t, y_t, logits, phase
         



X_train, y_train, X_test, y_test = getData()
m = y_train.shape[0]
n_y = y_train[0].shape[0]

num_complete_minibatches = int(math.floor(m/mini_batch_size))

global_step = tf.placeholder(tf.int32, name='global_step')
# get network
X_t, y_t, logits, phase = getMyNetwork(n_y)
# Training computation.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_t))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = cost  + reg_constant * sum(reg_losses)
# learning rate decay 
learning_rate = tf.train.exponential_decay(starter_learning_rate, num_epochs, epoch_learning_rate_decay, 0.96, staircase=True)
# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
# Ensures that we execute the update_ops before performing the train_step
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# Initialize all the variables
init = tf.global_variables_initializer()
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

def prepare_mini_batch(k):
	first = k * mini_batch_size
	last = min((k + 1) * mini_batch_size, m)
	X_train_b = np.asarray(X_train[first : last].tolist(), dtype=np.float32)
	y_train_b = np.asarray(y_train[first : last].tolist(), dtype=np.float32)
	return X_train_b,y_train_b




# Ensures that we execute the update_ops before performing the train_step
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_t, 1), tf.argmax(logits, 1)),'float32'))


previous_cost = 9999.0
best_test_accuracy = 0.0

with tf.Session() as sess:
# Run the initialization
	tmps1=time.clock()
	# Before training, run this:
	tflearn.is_training(True, session=sess)
	if os.path.isfile(TRAINEDMODEL + '.index'):
		print "parameters loaded"
		saver.restore(sess, TRAINEDMODEL)
	else:
		print "parameters initialized"
		sess.run(init)
	tt = time.time()
	for epoch in range(num_epochs):
		tmps1=time.clock()
		epoch_cost = 0.0
		for k in range(num_complete_minibatches + 1):
			X_train_b, y_train_b = prepare_mini_batch(k)
			_ , minibatch_cost = sess.run([optimizer, loss], feed_dict={X_t: X_train_b, y_t: y_train_b,phase: 1, global_step: epoch})

		        epoch_cost += minibatch_cost * y_train_b.shape[0]/ m   
		tmps2=time.clock()
  			
  		if epoch % epoch_control == 0:
  			print("Cost mean epoch %i: %f" % (epoch, epoch_cost))
  			print "execution time epoch = %f" %(tmps2-tmps1)
  			print "total execution time = %f" %(time.time() - tt)
  			test_accuracy = accuracy.eval({X_t: X_test, y_t: y_test, phase: 0})
  			print("Train Accuracy:", accuracy.eval({X_t: X_train, y_t: y_train, phase: 0}))
  			print ("Test Accuracy:", test_accuracy)
  			print("\n")
  			if test_accuracy > best_test_accuracy:
  			    best_test_accuracy = test_accuracy
  			    save_path = saver.save(sess, TRAINEDMODEL)
print("Best test Accuracy:", test_accuracy) 
