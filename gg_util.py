"""
Created on Thu Dec 14 10:06:20 2017

@author: guillaume
"""

from pathlib import Path
import numpy as np
import pandas as pd

def input_path():
    """
    output:
        "Kaggle input" folder location as string
    """
    my_dir = "input/"
    if Path.is_dir(Path(my_dir)):
        return my_dir
    elif Path.is_dir(Path("../" + my_dir)):
        return "../" + my_dir
    return None

def has_na(data_frame):
    """
    input:
        data_frame, pd.DataFrame
    output:
        boolean, if data_frame has at least a NA value
    """
    assert data_frame is not None
    return data_frame.isnull().values.any()

def print_head(data, nb_lines=10):
    """
    input:
        data, pd.DataFrame or np.ndarray
        nb_lines nb of lines to display
    output:
        string, first nb_lines of data_frame
    """
    #FIXME use isinstance instead of type
    #assert() n is integer and >=0
    assert data is not None
    if type(data) is pd.DataFrame:
        print(data.head(nb_lines))
    elif type(data) is np.ndarray:
        #ndarray is supposed to be 2 dimensional
        print(data.T[:nb_lines])
#        for row in data:
#            print("")

def iff(if_arg, arg1, arg2):
    """
    Just because it's convenient
    """
    if if_arg:
        return arg1
    return arg2

#print_head(np.random.randn(2,20))
