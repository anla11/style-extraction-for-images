#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:36:04 2017

@author: anla
"""
import numpy as np
from random import shuffle
import re 
import pandas as pd

from datapath_process import get_list_dir, get_list_filedir, get_list_filename

def shuffle_data(data_folder, path):
    folder_names = get_list_dir(data_folder)
    print (len(folder_names))
    imgs_name = []
    imgs_type = []
    for folder in folder_names:
        names = get_list_filename(folder_names[0])
        imgs_name = imgs_name + names
    shuffle(imgs_name)
    imgs_path = ['%s/%s' % (path, x) for x in imgs_name]
    
    imgs_type = [("tv_wide" if x.startswith("tv_wide") else "tv_small") for x in imgs_name]     

    data_df = pd.DataFrame({'img_path':imgs_path, 'img_type':imgs_type})    
    train_df = pd.DataFrame({'img_path':imgs_path[:int (0.8 * len(imgs_path))], 'img_type':imgs_type[:int (0.8 * len(imgs_type))]})
    test_df = pd.DataFrame({'img_path':imgs_path[int (0.8 * len(imgs_path)):], 'img_type':imgs_type[int (0.8 * len(imgs_type)):]})
    
    print train_df
    print test_df
    train_df.to_csv("files/fpt_train.csv")
    test_df.to_csv("files/fpt_test.csv")
    data_df.to_csv("files/fpt_data.csv")
    
    index = np.where(train_df['img_type'] == 'tv_wide')[0]
    print len(train_df), len(test_df)
    print len(index)
    sample_df = train_df.loc[index]
    print len(data_df), len(sample_df)
    sample_df.to_csv("files/fpt_tv_wide.csv")
    
path = 'data/fpt/highlights_sample'
shuffle_data('../../%s' % path, path)






