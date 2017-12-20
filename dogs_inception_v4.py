"""
https://github.com/tensorflow/models/tree/master/research/slim#Install
Il faut installer inception_v4.cpt a partir de github
je l'ai installe dans tools

cd $HOME/kaggle/tools
git clone https://github.com/tensorflow/models/

puis

cd models/research/slim
python setup build
python setup install

ne pas oublier de recuperer
http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
et de le dezipper dans tools/inception_v4/

"""
"""
Structure
kaggle/tools/models/research/slim
kaggle/tools/inception_v4/inception_v4.ckpt
kaggle/./data/labels.csv
kaggle/./data/train
kaggle/./data/test
kaggle/./cache/inception_v4
"""

import sys
import os
import gc
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
import types
import tensorflow as tf
import math
from nets import inception
from tensorflow.python import pywrap_tensorflow
from nets import inception_v4

sys.path.insert(0,'/root/kaggle/tools/models')
slim = tf.contrib.slim

IMGPATH = './data/train/'
FDATA = './data/labels.csv'
FILECACHE = './cache/inception_v4/cache_all_data'

"""
Parameters
"""
image_width = 299
num_channels = 3

def image_to_array(path,filename):
    """
    Transforme une image en numpy array
    """
    infile = os.path.join(path,filename[0] +'.jpg')
    if (os.path.isfile(infile)):
        try:
            image = np.array(ndimage.imread(infile, flatten=False,mode="RGB"))
            if (image.shape[0] * image.shape[1] * image.shape[2] < 400000000000):
                my_image = scipy.misc.imresize(image, size=(image_width,image_width))
                return(my_image)
            else:
                print(my_image)    
        except (IOError,MemoryError):
            print("memoryError")
    else:
        print(infile)
    return False    
        
def prepare_datas(path,filenames,breeds = False):
    """
       Prepare un tuple au format (X,y)
       X est le tableau des images
       y est le tableau des labels
    """        
    m = filenames.shape[0]
    X = []
    y = []
    label = []
    for l in range(m):
        img = image_to_array(path,filenames[l])
        if type(img) != types.BooleanType:
            X.append(img)
            if (type(breeds) != types.BooleanType):
                y.append(breeds[l])
    return (X, y)

def create_cache():
    """
       Cree le cache au format (X,y) et l'enregistre
    """
    df = pd.read_csv(FDATA)
    df = df[["id","breed"]]
    df = df.dropna()
    breeds = df["breed"].as_matrix()
    filenames = df[["id"]].as_matrix()
    data = prepare_datas(IMGPATH,filenames,breeds)
    np.save( FILECACHE, data )
    return data

def get_data():
    """
       Renvoie les donnees de base au format (X,y)
    """    
    if os.path.isfile(FILECACHE + '.npy'):
        data = np.load(FILECACHE + '.npy')
    else:
        data = create_cache()
    return data
    
def getInceptionv4SelectedLayer(layer):
    """
     get pretained inception v4 selected layer
     Renvoie le tenseur d'entree X_t et le tenseur de sortie selectionne final_endpoint
    """
    X_t = tf.placeholder(shape = [ None, image_width,image_width,3],dtype = tf.float32, name = "X_t")
    # FLATTEN
    # final_endpoint_flat = tf.contrib.layers.flatten(final_endpoint)
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        final_endpoint, end_points = inception_v4.inception_v4_base(X_t, final_endpoint=layer)

    """
    final_endpoint, end_points = inception_v4.inception_v4(X_t,num_classes=1001,is_training=False)
    for ep in end_points:
        print ep, end_points[ep]
    """
    # Affiche les infos sur le layer
    print( layer, end_points[layer])
        # final_endpoint_flat = tf.contrib.layers.flatten(final_endpoint)
    return X_t,final_endpoint

def getLayerOutputFromPretrainedInceptionv4(layer,data_image):
    """
        layer = le layer inception v4 attendu
        data_image = les donnees images RGB sous forme numpy ( m x 299 x 299 x 3 )
        return les donnees transformes par inception v4 jusqu'au layer X
    
    """    
    TRAINEDMODEL = '../tools/inception_v4/inception_v4.ckpt'
    mini_batch_size = 128
    m = data_image.shape[0]
    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    X_t,selected_layer = getInceptionv4SelectedLayer(layer)

    def prepare_mini_batch(k):
        batch_begin = k * mini_batch_size
        batch_end = min((k + 1) * mini_batch_size, m)
        X_batch = np.asarray(data_image[batch_begin : (k + 1) * mini_batch_size].tolist(), dtype=np.float32)
        return X_batch

    my_array = np.array([])
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Add ops to save and restore all the variables.
    # saver = tf.train.Saver(tf.global_variables()) # , write_version=tf.train.SaverDef.V1
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        """

        recupere les variables pre-entraines necessaires pour le layer selectionne depuis inception_v4.ckpt

        """
        variables=tf.global_variables()
        reader=pywrap_tensorflow.NewCheckpointReader(TRAINEDMODEL)
        var_keep_dic=reader.get_variable_to_shape_map()
        variables_to_restore = []
        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                variables_to_restore.append(v)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, TRAINEDMODEL)
        for k in range(num_complete_minibatches + 1):
            X_batch = prepare_mini_batch(k)
            result = sess.run(selected_layer, {X_t: X_batch})
            if k == 0: 
                my_array = result
            else:
                my_array = np.concatenate((my_array,result), axis=0)
        tf.reset_default_graph()
    return  my_array

def get_data_from_pretrained_Inceptionv4(layer):
	INCEPTION_X_FILE = './cache/inception_v4/X_inception_v4_'+ layer + '.npy'
	LABEL_FILE = './cache/inception_v4/y_label.npy'
	if (os.path.isfile(INCEPTION_X_FILE)  or os.path.isfile('./cache/inception_v4/X_alex_'+ layer + '.npy')):
		X = np.load(INCEPTION_X_FILE)
		y = np.load(LABEL_FILE)
	else:
		data = get_data()
		y = data[1]
		X = getLayerOutputFromPretrainedInceptionv4(layer,np.array(data[0]))
		np.save( LABEL_FILE, y )
		np.save( INCEPTION_X_FILE, X )
	return X,y

FILENAMETESTCACHE = './cache/inception_v4/cache_test_filename'

def create_test_data():
	IMAGETESTPATH = './data/test/'
	filenames_test = np.expand_dims(np.array([s.replace(".jpg","") for s in os.listdir(IMAGETESTPATH)]), axis=1)
	images_test,_ = prepare_datas(IMAGETESTPATH,filenames_test)
	return filenames_test,np.array(images_test)

def get_test_data():
	FILETESTCACHE = './cache/inception_v4/cache_test_data'
	if (os.path.isfile(FILETESTCACHE + '.npy')):
		data = np.load(FILETESTCACHE + '.npy')
		filenames = np.load(FILENAMETESTCACHE + '.npy')
	else:
		filenames,data = create_test_data()
		np.save(FILENAMETESTCACHE,filenames)
		np.save(FILETESTCACHE,data)
	return filenames,data

def get_test_data_from_pretrained_Inceptionv4(layer):
	INCEPTION_X_TEST_FILE = './cache/inception_v4/X_inception_v4_test_'+ layer + '.npy'
	if (os.path.isfile(INCEPTION_X_TEST_FILE)):
		X_test = np.load(INCEPTION_X_TEST_FILE)
		filenames = np.load(FILENAMETESTCACHE + '.npy')
	else:
		filenames,images = get_test_data()
		X_test = getLayerOutputFromPretrainedInceptionv4(layer,images)
		np.save( INCEPTION_X_TEST_FILE, X_test)
	return filenames,X_test	

if __name__ == "__main__":
    X,y = get_data_from_pretrained_Inceptionv4('Mixed_7d')
    del X
    del y
    gc.collect()
    filenames_test,X_test = get_test_data_from_pretrained_Inceptionv4('Mixed_7d')

