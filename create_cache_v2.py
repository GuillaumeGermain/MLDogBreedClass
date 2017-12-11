import os
import Image
import pandas as pd
import numpy as np
#import math
from sklearn.model_selection import train_test_split
import scipy
from scipy import ndimage
import types


Image.MAX_IMAGE_PIXELS = 100_000_000_000

IMGPATH = '/root/kaggle/dogs/train/'
FDATA = '/root/kaggle/dogs/labels.csv'
OUTFILETEST = '/root/kaggle/dogs/cache_test'
OUTFILETRAIN = '/root/kaggle/dogs/cache_train'

mini_batch_size = 124
image_size = 227
num_channels = 3




def mini_batch_image(filename):
	infile = os.path.join(IMGPATH,filename[0] +'.jpg')
	if (os.path.isfile(infile)):
		try:
			image = np.array(ndimage.imread(infile, flatten=False,mode="RGB"))
			if (image.shape[0] * image.shape[1] * image.shape[2] < 400000000000):
				my_image = scipy.misc.imresize(image, size=(image_size,image_size))
			else:
				print(my_image)
				my_image = False
			return my_image
		except (IOError,MemoryError):
			print("memoryError")
			return False
	else:
		print(infile)
		return False



def prepare_images(mini_batch_X, mini_batch_Y):
	L = mini_batch_Y.shape[0]
	X = []
	y = []
	label = []
	for l in range(0,L):
		img = mini_batch_image(mini_batch_X[l])
		if type(img) != types.BooleanType:
			X.append(img)
			label.append(mini_batch_Y[l,0])
			y.append(mini_batch_Y[l,1:])
	return (X, label, y)


def prepare_data( X, y, file):
	m1 = X.shape[0]
	data = prepare_images(X,y)
	np.save( file, data )



# Step 1: Shuffle (X, Y)

# permutation = list(np.random.permutation(m))
# shuffled_X = filenames[permutation,:]
# shuffled_Y = labels[permutation,:]

df = pd.read_csv(FDATA)
df = df[["id","breed"]]
df = df.dropna()
dum = pd.get_dummies(df["breed"]).as_matrix()
breeds = df["breed"].as_matrix()
breeds = np.expand_dims(breeds, axis=1)
print(dum.shape)
print(breeds.shape)
labels = np.append(breeds,dum,axis=1)
filenames = df[["id"]].as_matrix()

# m = labels.shape[0]                  # number of training examples
mini_batches = []

print(df["id"].shape)
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split( filenames, labels, test_size=0.2)

print(X_test.shape)


prepare_data( X_train, y_train, OUTFILETRAIN)
prepare_data( X_test, y_test, OUTFILETEST)


