#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:00:44 2017

@author: anla
"""
import sys, getopt

import pandas as pd
import numpy as np

from cluster import read
from recommendation.recommend import run
from style_feature import Style_Feature

import matplotlib.pyplot as plt
import cv2 

#resource
img_dataset = '/home/anla/Source/data/images/sample.csv' #a files contains path of images in dataset
input_path = 'recommend/input.csv' #a path where contains results
output_path = 'recommend/gist-svm_32cluster'

glb_style_obj, glb_n_cluster, glb_cluster_model = None, None, None
glb_processgist_model = None

def     init(f_gist, f_lab, processgist_model, n_cluster, cluster_model):
    global glb_n_cluster
    glb_n_cluster = n_cluster
    global glb_cluster_model
    glb_cluster_model = cluster_model
    global glb_processgist_model
    glb_processgist_model = processgist_model
    global glb_style_obj
    glb_style_obj = Style_Feature(f_lab, f_gist, processgist_model)    
    
def view_img_2(imgpath, clusters, name):
    row = 0       
    for t in range(len(clusters)):
        cluster = clusters[t]
        for i in range(0, len(cluster), 4):      
            for j in range(4):
                if i + j >= len(cluster):
                    break
                img = cv2.imread(imgpath[cluster[i+j]])
                cv2.imwrite('%s/Cluster_%d_img_%d.jpg' %(name, t+1, i+j), img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            row += 1

def userimg_description(user_clusters, users_data, name):
    cluster_num = []
    for i in range(1, glb_n_cluster+1):
        cluster_num.extend([i] * len(user_clusters[i-1]))

    img_path = []
    for i in range(len(user_clusters)):
        cluster = user_clusters[i]
        for t in range(len(cluster)):
            img_path.append(users_data[cluster[t]])
        
    res = pd.DataFrame({'img_path':img_path, 'cluster':cluster_num})
    res.to_csv(name)
    
def to_file(img_rank, cluster_tops, data_imgpath, name):
    cluster_num = []
    for i in range(1, glb_n_cluster+1):
        cluster_num.extend([i] * cluster_tops[i-1]) 

    sim = []
    rank = []
    img_path = []
    for i in range(len(cluster_tops)):
        top = cluster_tops[i]
        cluster = img_rank[i]
        for t in range(top):
            img_path.append(data_imgpath[cluster[t]['index']])
            sim.append(cluster[t]['sim'])
        rank.extend(list(range(1, top+1)))
        
    res = pd.DataFrame({'rank':rank, 'img_path':img_path, 'cluster':cluster_num, 'sim':sim})
    res.to_csv(name)

  
def recommend_1(top, data_imgpath, img_rank):
    cluster_tops = [min(top, len(cluster)) for cluster in img_rank]
    rec_list = np.array([[item['index'] for item in img_rank[i][:cluster_tops[i]]] for i in range(glb_n_cluster)])    

    to_file(img_rank, cluster_tops, data_imgpath, '%s/output_1.csv' % output_path)
    view_img_2(data_imgpath, rec_list, '%s/output_1' % output_path)
    
def recommend_2(total, data_imgpath, img_rank, cluster_ratio):
    cluster_tops = [min(len(img_rank[i]), int(round(cluster_ratio[i]['ratio'] * total))) for i in range(len(cluster_ratio))]
    print cluster_tops
    rec_list = np.array([[item['index'] for item in img_rank[i][:cluster_tops[i]]] for i in range(glb_n_cluster)])    
    to_file(img_rank, cluster_tops, data_imgpath, '%s/output_2.csv'% output_path)
    view_img_2(data_imgpath, rec_list, '%s/output_2'% output_path)

def     main(argv):
    f_gist, f_lab = 1, 0
    cluster_model = '/home/anla/Source/python/style_feature_extraction/cluster/movies/gist_svm/res_32/kmeans_svmgist.plk'
    processgist_model = '/home/anla/Source/python/style_feature_extraction/classification/sr_models/svm_sr_gist.pkl'

    init(f_gist = f_gist, f_lab = f_lab, n_cluster = 32, cluster_model = cluster_model, processgist_model = processgist_model)

    data_features, data_imgpath = read(img_dataset, glb_style_obj)
    user_features, user_imgpath = read(input_path, glb_style_obj)
    
    img_rank, cluster_ratio, user_cluster = run(data_features, user_features, glb_cluster_model, glb_cluster_model)
    userimg_description(user_cluster, user_imgpath, '%s/input_rank.csv' % output_path)
    view_img_2(user_imgpath, user_cluster, glb_n_cluster, '%s/input' % output_path)
    
    recommend_1(40, data_imgpath, img_rank)
    recommend_2(300, data_imgpath, img_rank, cluster_ratio)

    
if __name__ == "__main__":
   main(sys.argv[1:])
