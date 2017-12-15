"""
Created on Tue Dec 12 10:58:48 2017

@author: guillaume
"""


#Today quick playing with os and pandas framework
#from pathlib import Path
import pandas as pd
from gg_util import iff, input_path, has_na, print_head
import numpy as np

#Constants
INPUT_PATH = input_path()
assert INPUT_PATH is not None
IMG_PATH = INPUT_PATH + "train/"
IMG_LIST = INPUT_PATH + "labels.csv"
OUT_FILE_TEST = INPUT_PATH + "cache_test"
OUT_FILE_TRAIN = '/root/kaggle/dogs/cache_train'

#print("IMG_LIST", IMG_LIST)

#df: DataFrame
df = pd.read_csv(IMG_LIST)

#print_head(df, 10)
#print("df.shape", df.shape)
#print("type(df)", type(df))


#is there any NA value?
print("has df NA values?", iff(has_na(df), "Yes", "No"))

#Just to be double-sure for next usages of this file...
df.dropna()

#get_dummies converts variables with limited discrete values into numbers
#outputs dataframe
#as_matrix(): stores as np array
df_breeds = pd.get_dummies(df["breed"]).as_matrix()
print("breeds.shape", df_breeds.shape)
print_head(df_breeds)

#Add one 1-dim at the end
df_breeds = np.expand_dims(df_breeds, -1)
print("df_breeds.shape", df_breeds.shape)
print_head(df_breeds)

#Alain's way
#breeds = np.expand_dims(breeds, axis=1)
#labels = np.append(breeds, dum, axis=1)
#filenames = df[["id"]].as_matrix()


#OK this way it's managed with numpy arrays in a relatively complicated way
#let's do it with pandas

