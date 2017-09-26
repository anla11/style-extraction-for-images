from datapath_process import get_list_filename
import numpy as np
from random import shuffle
import re 
import pandas as pd


def shuffle_data(data_folder, path):
    names = [('%s/%s' % (path, str_name)) for str_name in get_list_filename("%s" % data_folder)]
    
    shuffle(names)
    labels = [re.split('[_ /]', name)[-2] for name in names]
    print names[0], labels[0]
    data_df = pd.DataFrame({'img_path':names, 'label':labels})
    train_df = pd.DataFrame({'img_path':names[:int (0.8 * len(names))], 'label':labels[:int (0.8 * len(names))]})
    test_df = pd.DataFrame({'img_path':names[int (0.8 * len(names)):], 'label':labels[int (0.8 * len(names)):]})
    print train_df.head()
    print test_df.head()
    print len(train_df), len(test_df)
    train_df.to_csv("files/sr_train.csv")
    test_df.to_csv("files/sr_test.csv")
    data_df.to_csv("files/sr_data.csv")

path = 'data/scence_recognition/scence_recognition_full/image'
shuffle_data('../../%s' % path, path)





