"""
Created on Tue Dec 12 10:58:48 2017

@author: guillaume
"""


#Today quick playing with os and pandas framework
from pathlib import Path
import pandas as pd



#print("os.name", os.name)
#print("getenv", os.get_exec_path())


def input_path():
    """
    output: "Kaggle input" folder location as string
    """
    my_dir = "input/"
    if Path.is_dir(Path(my_dir)):
        return my_dir
    elif Path.is_dir(Path("../" + my_dir)):
        return "../" + my_dir
    return None

def has_na(df):
    """
    input: df, pd.DataFrame
    output: boolean, if df has at least a NA value
    """
    assert df is not None
    return df.isnull().values.any()

def print_head(df, n):
    """
    input:
        df, pd.DataFrame
        n nb of lines to display
    output: boolean, if df has at least a NA value
    """
    #assert() n is integer and >=0
    assert(df is not None)
    print(df.head(n))

def iff(a,b,c):
    if a:
        return b
    else:
        return c

#Constants
INPUT_PATH = input_path()
IMG_PATH = INPUT_PATH + "train/"
IMG_LIST = INPUT_PATH + "labels.csv"
OUT_FILE_TEST = INPUT_PATH + "cache_test"
OUT_FILE_TRAIN = '/root/kaggle/dogs/cache_train'

#print("IMG_LIST", IMG_LIST)


data_file = pd.read_csv(IMG_LIST)

#print_head(data_file, 10)
#print("data_file.shape", data_file.shape)
#print("type(data_file)", type(data_file))


#is there any NA value?
print("has df NA values?", iff(has_na(data_file), "Yes", "No"))

#Just to be double-sure for next usages of this file...
data_file.dropna()





