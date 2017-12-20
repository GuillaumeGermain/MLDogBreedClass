import dogs_inception_v4
import train_nn_based_on_inception_v4
import numpy as np
import tensorflow as tf
import pandas as pd

TRAINEDMODEL = '/root/kaggle/dogs/model.ckpt'

filenames_test, X_test = dogs_inception_v4.get_test_data_from_pretrained_Inceptionv4('Mixed_7d')
breeds = np.load(FILECOLUMNS)
X_t, y_t, logits, phase = train_nn_based_on_inception_v4.getMyNetwork(size(breeds))
predictions = tf.nn.softmax(logits)
saver = tf.train.Saver()
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, TRAINEDMODEL)
    proba = predictions.eval({X_t: X_test,phase: 0})
filenames = np.array(filenames_test).T[0]
df = pd.DataFrame(data=proba,               # values
                  index=filenames,          # 1st column as index
                  columns=breeds)     # 1st row as the column names
df.index.name = "id"
df.to_csv('./data/inception_submission.csv')
