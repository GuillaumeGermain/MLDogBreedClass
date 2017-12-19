"""
    Main source code for the Kaggle dog breed classification competition
    
    AlexNet
    Implementation of AlexNet 
    The chosen model has 8 layers
    We take the 7 first ones, and we complete by a MultinomialNB function for the results
    
    Inception v3: coming soon
    
    Folder structure:
    /data                 data folder with labels.csv
    /data/train           training data
    /data/test            test data
    /tools                models
    /tools/alexnet        alexnet models and weights
    /tools/inception_v3   inception v3 model and weights
    /result               model results folder

    The data, models and result are not shared on GitHub
    
"""
import os
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
#import types
import tensorflow as tf
import math
from gg_util import INPUT_PATH

"""
External Files
"""
IMG_PATH   = INPUT_PATH + 'train/'
FDATA      = INPUT_PATH + 'labels.csv'
FILE_CACHE = INPUT_PATH + 'cache/alexnet_ML/cache_all_data'

"""
Parameters
"""
IMAGE_WIDTH = 227
NUM_CHANNELS = 3

def image_to_array(path, filename):
    """
    Transforme une image en numpy array
    """
    infile = os.path.join(path, filename[0] + '.jpg')
    if os.path.isfile(infile):
        try:
            image = np.array(ndimage.imread(infile, flatten=False, mode="RGB"))
            if image.shape[0] * image.shape[1] * image.shape[2] < 400_000_000_000:
                my_image = scipy.misc.imresize(image, size=(IMAGE_WIDTH, IMAGE_WIDTH))
                return my_image
            else:
                print(my_image)
        except (IOError, MemoryError):
            print("memoryError")
    else:
        print(infile)
    return False

def prepare_data(path, filenames, breeds=None):
    """
       Prepare un tuple au format (X,y)
       X est le tableau des images
       y est le tableau des labels
    """
    m = filenames.shape[0]
    X = []
    y = []
    #label = []
    for l in range(m):
        img = image_to_array(path, filenames[l])
        if type(img) != bool:
            X.append(img)
            if breeds != None:
                y.append(breeds[l])
    return (X, y)

def create_cache():
    """
       Cree le cache au format (X,y) et l'enregistre
    """
    df = pd.read_csv(FDATA)
    df = df[["id", "breed"]]
    df = df.dropna()
    breeds = df["breed"].as_matrix()
    filenames = df[["id"]].as_matrix()
    data = prepare_data(IMG_PATH, filenames, breeds)
    np.save(FILE_CACHE, data)
    return data

def get_data():
    """
       Renvoie les donnees au format (X,y)
    """
    if os.path.isfile(FILE_CACHE + '.npy'):
        data = np.load(FILE_CACHE + '.npy')
    else:
        data = create_cache()
    return data

def conv(input_, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''
        From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input_.get_shape()[-1]
    assert c_i%group == 0
    assert c_o%group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group == 1:
        conv = convolve(input_, kernel)
    else:
        input_groups = tf.split(input_, group, 3)   #tf.split(3, group, input_)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def getAlexnetSelectedLayer(layer):
    """
        Get pretained alexnet selected layer
    """
    PRETRAINED_AXELNET = '../tools/alexnet/bvlc_alexnet.npy'
    X_t = tf.placeholder(shape = [None, IMAGE_WIDTH, IMAGE_WIDTH, 3], dtype=tf.float32, name="X_t")
    #pre trained Constants
    net_data = np.load(PRETRAINED_AXELNET).item()

    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.constant(net_data["conv1"][0])
    conv1b = tf.constant(net_data["conv1"][1])
    conv1_in = conv(X_t, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, 
                              ksize=[1, k_h, k_w, 1], 
                              strides=[1, s_h, s_w, 1], 
                              padding=padding)

    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.constant(net_data["conv2"][0])
    conv2b = tf.constant(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, 
                              ksize=[1, k_h, k_w, 1], 
                              strides=[1, s_h, s_w, 1], 
                              padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.constant(net_data["conv3"][0])
    conv3b = tf.constant(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.constant(net_data["conv4"][0])
    conv4b = tf.constant(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.constant(net_data["conv5"][0])
    conv5b = tf.constant(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.constant(net_data["fc6"][0])
    fc6b = tf.constant(net_data["fc6"][1])
    #fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(tf.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, tf.cast((tf.reduce_prod(maxpool5.get_shape()[1:])), tf.int32)]), fc6W, fc6b)

    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.constant(net_data["fc7"][0])
    fc7b = tf.constant(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    #fc8
    #fc(1000, relu=False, name='fc8')
    # fc8W = tf.constant(net_data["fc8"][0])
    # fc8b = tf.constant(net_data["fc8"][1])
    if layer == "fc7":
        selected_layer = fc7
    elif layer == "fc6":
        selected_layer = fc6
#    else:
#        selected_layer = fc8
    return X_t, selected_layer


def getLayerOutputFromPretrainedAlexnet(layer, data_image):
    mini_batch_size = 128
    m = data_image.shape[0]
    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    X_t, selected_layer = getAlexnetSelectedLayer(layer)
    # Initialize all the variables
    # init = tf.global_variables_initializer()
    
#    def prepare_mini_batch(k):
##        first = k * mini_batch_size
##        last = min((k + 1) * mini_batch_size, m)
#        X_batch = np.asarray(data_image[k * mini_batch_size : (k + 1) * mini_batch_size].tolist(), dtype=np.float32)
#        return X_batch
    
    my_array = np.array([])
    with tf.Session() as sess:
    # Run the initialization
        for k in range(num_complete_minibatches + 1):
#            X_batch = prepare_mini_batch(k)
            X_batch = np.asarray(data_image[k * mini_batch_size : (k+1) * mini_batch_size].tolist(), 
             dtype=np.float32)
            result = sess.run(selected_layer, {X_t: X_batch})
            if k == 0:
                my_array = result
            else:
                my_array = np.concatenate((my_array, result), axis=0)
    return  my_array


def get_data_from_pretrained_alexnet(layer):
    ALEX_X_FILE = './tools/alexnet/X_alex_'+ layer + '.npy'
    LABEL_FILE = './tools/alexnet/y_label.npy'
    if os.path.isfile(ALEX_X_FILE):
        X = np.load(ALEX_X_FILE)
        y = np.load(LABEL_FILE)
    else:
        data = get_data()
        y = data[1]
        X = getLayerOutputFromPretrainedAlexnet(layer, data[0])
        np.save(LABEL_FILE, y)
        np.save(ALEX_X_FILE, X)
    return X, y


FILENAME_TEST_CACHE = './tools/alexnet/cache_test_filename'

def create_test_data():
    IMAGETESTPATH = './data/test/'
    filenames_test = np.expand_dims(np.array([s.replace(".jpg", "") for s in os.listdir(IMAGETESTPATH)]), axis=1)
    images_test, _ = prepare_data(IMAGETESTPATH, filenames_test)
    return filenames_test, np.array(images_test)

def get_test_data():
    FILETESTCACHE = './tools/alexnet/cache_test_data'
    if os.path.isfile(FILETESTCACHE + '.npy'):
        data = np.load(FILETESTCACHE + '.npy')
        filenames = np.load(FILENAME_TEST_CACHE + '.npy')
    else:
        filenames, data = create_test_data()
        np.save(FILENAME_TEST_CACHE, filenames)
        np.save(FILETESTCACHE, data)
    return filenames, data

def get_test_from_pretrained_alexnet(layer):
    ALEX_X_TEST_FILE = './tools/alexnet/X_alex_test_'+ layer + '.npy'
    if os.path.isfile(ALEX_X_TEST_FILE):
        X_test = np.load(ALEX_X_TEST_FILE)
        filenames = np.load(FILENAME_TEST_CACHE + '.npy')
    else:
        filenames, images = get_test_data()
        X_test = getLayerOutputFromPretrainedAlexnet(layer, images)
        np.save(ALEX_X_TEST_FILE, X_test)
    return filenames, X_test


from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import cross_val_predict
#from sklearn import metrics
X, y = get_data_from_pretrained_alexnet('fc7')


#Un petit Multinomial sur la fin
#TODO ajouter une couche et un softmax?
clf = MultinomialNB(alpha=.01)
clf.fit(X, y)
filenames_test, X_test = get_test_from_pretrained_alexnet('fc7')
proba = clf.predict_proba(X_test)
filenames = np.array(filenames_test).T[0]
print(filenames)
df = pd.DataFrame(data=proba,               # values
                  index=filenames,          # 1st column as index
                  columns=clf.classes_)     # 1st row as the column names
df.index.name = "id"
df.to_csv('./tools/alexnet/submission.csv')
