#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:32:28 2017

@author: anla
"""
import numpy  as np
import sys
sys.path.append(".")

from style_feature import Style_Feature

from scipy.spatial.distance import euclidean
from operator import itemgetter
from sklearn.externals import joblib

#global variables - set as hyper parameters

  
def   cal_rank(data_features, data_label, user_features, user_label, n_cluster):
    d = np.zeros_like(data_label).astype(float)
    user_cluster = []
    for label in range(n_cluster):
        user_cluster.append(list(np.where(np.array(user_label) == label)[0]))
    
    for j in range(len(data_label)):
        label = data_label[j]
        for i in user_cluster[label]:
            d[j] += euclidean(user_features[i], data_features[j]) / len(user_cluster[label])
    rank = []
    for t in range(n_cluster):
        rank.append([])
    for i in range(len(data_label)):
        rank[data_label[i]].append({'index': i, 'sim':d[i]})
    for t in range(n_cluster):
        rank[t] = sorted(rank[t], key=itemgetter('sim'), reverse = False)
    return rank, user_cluster

def run(data_features, user_features, cluster_model, n_cluster):
    #cal cluster
    kmeans = joblib.load(cluster_model)   
    data_label = kmeans.predict(data_features)
    user_label = kmeans.predict(user_features)

    #cal average similarity of each image to user_img and rank them 
    img_rank, user_cluster = cal_rank(data_features, data_label, user_features, user_label, n_cluster)
    
    # cal ratio cluster in clusters_user 
    ratio = [len(np.where(np.array(user_label) == i)[0]) *1.0/len(user_label) for i in range(n_cluster)]
    cluster_ratio = sorted([{'cluster':i, 'ratio':ratio[i]} for i in range(n_cluster)], key=itemgetter('ratio'), reverse = True)
    return img_rank, cluster_ratio, user_cluster
  
        
