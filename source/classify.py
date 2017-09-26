#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 07:47:48 2017

@author: anla
"""

import numpy as np
import pandas as pd 
import os
import sys, getopt

from classification.classify import classfify_svm
from sklearn.externals import joblib
from style_feature import Style_Feature
from read_path import load_img


def     read_data(data_path, style_obj):
    print ('Reading %s' % (data_path))
    df = pd.read_csv(data_path)
    index = list(range(len(df)))
    np.random.shuffle(index)

    feature = np.array([style_obj.get_feature(load_img(df['img_path'][i])) for i in index])
    label = np.array(df.iloc[index]['label'])
    drop_idx = [i for i in range(len(feature)) if feature[i] is None]

    if len(drop_idx) > 0:
        feature = np.delete(feature,drop_idx,axis=0)
        label = np.delete(label,drop_idx,axis=0)
        
    data = np.zeros((len(feature), len(feature[0])))
    for i in range(len(label)):
        for j in range(len(feature[i])):
            data[i][j] = feature[i][j]
    print (data.shape)
        
    return data, label

def    run(X, y, mode = 'train', max_iter = -1, modelpath = 'model.pkl'):
    if mode == 'train':
        print ('Training')
        clf = classfify_svm(X, y, max_iter)
        print ('Save model as %s ' % modelpath)
        joblib.dump(clf, modelpath, compress=1)
    else:
        clf = joblib.load(modelpath)
    pred = clf.predict(X)
    print ('Accuracy: ', np.sum(pred == y) * 1.0/len(pred))
    

def     main(argv):
    def     instruction():
        print '     scence_recognition.py --train --in=<input_folder> --iter=<max_iter> --f_lab=<True/False> --f_gist=<1/0> --model=<save_path>'
        print 'or   scence_recognition.py --test --in=<input_folder> --f_lab=<1/0> --f_gist=<1/0> --model=<load_path>'
        
    mode, data_path, max_iter, f_lab, f_gist, modelpath = False, '../../example/scence_recognition/scence_recognition_full/train.csv ', 1000, 0, 1, ''
    try:
        opts, args = getopt.getopt(argv, "h:t:in:it:fl:fg:m", ["help", "train", "test", "in=", "iter=", "f_lab=", "f_gist=", "model="])
    except getopt.GetoptError:
        sys.exit(2)
    opt, arg = opts[0]
    
    
    if opts in ("-h", "--help"):
        instruction()
        sys.exit()
    elif opt == '--train' or opt == '--test':
        if opt == '--train':
            mode = 'train'
        else:
            mode = 'test'       
        for opt, arg in opts[1:]:
            if opt == "--in":
                data_path = arg
            elif opt == '--iter':
                max_iter = int(arg)
            elif opt == '--f_lab':
                f_lab = int(arg)
            elif opt == "--f_gist":
                f_gist = int(arg)
            elif opt in ("--model"):
                modelpath = arg

    if os.path.exists(data_path) == False:
        print "No such file %s." % (data_path)
        return
    if mode == 'test' and os.path.exists(modelpath) == False:
        print "No such file %s." % (modelpath)
        return

    style_obj = Style_Feature(lab = f_lab, gist = f_gist, gist_processmodel = None)    
    
    features, label = read_data(data_path, style_obj)
    run(features, label , mode, max_iter, modelpath)    
        

if __name__ == "__main__":
   main(sys.argv[1:])
    

#max_iter, f_lab, f_gist = 1000, 1, 1
#data_path = '/mnt/e/Workspace/GitSpace/style_feature_extraction/data/scence_recognition_full/test.csv'
#model = 'svm_sr_gist-lab.pkl'
#
#style_obj = Style_Feature(lab = f_lab, gist = f_gist)    
#features, label = read_data(data_path, style_obj)
#
#
#run(data, label, 'test', max_iter = 1000, modelpath = model) 

'''

Save model as svm_sr_lab.pkl 
Train Accuracy:  0.518604651163
Test Accuracy:  0.418215613383

Save model as svm_sr_gist.pkl 
Train Accuracy:  0.928837209302
Test Accuracy:  0.862453531599

Save model as svm_sr_gist-lab.pkl 
Train Accuracy:  0.951162790698
Test Accuracy: 0.884758364312
'''



'''
python scence_recognition.py --train \
--in='../../example/scence_recognition/scence_recognition_full/train.csv' \
--iter=1000 --f_lab=1 --f_gist=1 \
--model='scence_recognition/models/svm_sr_gist-lab.pkl'
'''