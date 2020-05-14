#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:06:21 2017

@author: anla
"""
import numpy as np
import pandas as pd

import re
import os
import shutil 
import sys, getopt

from style_feature import Style_Feature
from read_path import load_img
from clustering.visualize import visualize
from clustering.clustering import get_kmeans, cal_distance

from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PATH_ROOT_PROJ = '../'


def     img_visualize(n_cluster, cluster_folder):
    for i in range(n_cluster):
        filename = '%s/Cluster_%03d.txt' % (cluster_folder, i)
        if os.stat(filename).st_size == 0:
            continue
        cluster_i = pd.read_csv(filename, header = None)     
       
        folder = '%s/cluster_%03d' % (cluster_folder, i)
        
        if os.path.exists(folder):    
            shutil.rmtree(folder)
        os.makedirs(folder)                 
        
        for j in cluster_i[0]:
            shutil.copyfile(j, '%s/%s' % (folder, re.split('/', j)[-1]))        

def     to_file(clusters, clusters_folder, img_paths, visualize_mode):
    for i in range(len(clusters)):
        cluster_img = [img_paths[j] for j in clusters[i]]
        np.savetxt('%s/Cluster_%03d.txt' % (clusters_folder, i), cluster_img, delimiter = " ", fmt = "%s")

    if visualize_mode:
        img_visualize(len(clusters), clusters_folder)
    
def     read(data_path, style_obj):
    df = None
    print "Load %s" % data_path
    df = pd.read_csv(data_path)    
    imgs = [load_img('%s/%s' % (PATH_ROOT_PROJ, img_path)) for img_path in df['img_path'] if os.path.exists(img_path)]
    imgs = [img for img in imgs if img is not None]
    features = np.array([style_obj.get_feature(img) for img in imgs])    
    drop_idx = [i for i in range(len(features)) if features[i] is None]
    features = np.delete(features,drop_idx,axis=0)
    
    data = np.zeros((len(features), len(features[0])))
    for i in range(len(features)):
        for j in range(len(features[i])):
            data[i][j] = features[i][j]
    print (data.shape)
        
    return data, df['img_path']

def     run(mode, features, img_path, cluster_folder, n_clusters, cluster_model = 'kmeans.pkl', visualize_mode = True):
    if os.path.exists(cluster_folder) == False:
        print "No such directory %s." % (cluster_folder)
        return
    if mode == 'test' and os.path.exists(cluster_model) == False:
        print "No such file %s." % (cluster_model)
        return
    
    model, clusters = get_kmeans(features, n_clusters, cluster_model, mode)
    to_file(clusters, cluster_folder, img_path, visualize_mode)

    P = cal_distance(features)
    sil_coef_avg = silhouette_score(features, model.labels_, metric='euclidean')
    print("For n_clusters={}, Average Silhouette Coefficient is {}".format(n_clusters, sil_coef_avg))
    
    visualize(features, P, model.labels_, n_clusters, cluster_folder)    

    if mode == 'train':
        joblib.dump(model, "%s/%s" % (cluster_folder, cluster_model))
        print 'Save model as %s/%s' % (cluster_folder, cluster_model)
    
def     main(argv):
    def     instruction():
        print '     cluster.py --train --in=<data_path> --out=<cluster_folder> --n_clusters=<n_clusters> --f_lab=<1/0> --f_gist=<1/0> --visualize=<1/0>'
        print 'or   cluster.py --test --in=<data_path> --out=<cluster_folder>  --n_clusters=<n_clusters> --model=<load_path> --f_lab=<1/0> --f_gist=<1/0> --visualize=<1/0>'
        print 'or   cluster.py --visualize --n_cluster=<n_cluster> --out=<cluster_folder>'
        
    data_path = '/home/anla/Source/data/images/sample.csv'
    mode = 'train'
    n_clusters = 8
    cluster_folder = '/home/anla/Source/python/style_feature_extraction/cluster/movies/gist/res3_'
    f_lab, f_gist = 0, 1
    cluster_model = 'kmeans.plk'
    processed_model = None
    visualize_mode = True
    
    try:
        opts, args = getopt.getopt(argv, "h:t:in:out:n:fl:fg:v:m", ["help", "train", "test", "visualize", "in=", "out=", "n_clusters=", "f_lab=", "f_gist=", "visualize=", "model="])
    except getopt.GetoptError:
        sys.exit(2)
    opt, arg = opts[0]
    
    if opts in ("-h", "--help"):
        instruction()
        sys.exit()
    elif opt in ("--train", "--test", "--visualize"):
        if opt == "visualize":
            mode = 'visualize'
        else:
            if opt == "--train":
                mode = 'train'
            else:
                mode = 'test'
            
        for opt, arg in opts[1:]:
            if opt == "--in":
                data_path = arg
            elif opt == '--out':
                cluster_folder = arg
            elif opt == '--f_lab':
                f_lab = int(arg)
            elif opt == "--f_gist":
                f_gist = int(arg)
            elif opt == "--n_clusters":
                n_clusters = int(arg)
            elif opt == "--visualize":
                visualize_mode = arg
            elif opt == "--model":
                cluster_model = arg
    
    if mode == 'visualize':
        img_visualize(n_clusters, cluster_folder)
    style_obj = Style_Feature(lab = f_lab, gist = f_gist, gist_processmodel = processed_model)
    features, img_path = read(data_path, style_obj)    
    run(mode, features, img_path, cluster_folder, n_clusters,  cluster_model, visualize_mode)
    
if __name__ == "__main__":
   main(sys.argv[1:])




'''
 python cluster.py --train --in="/home/anla/Source/data/images/sample.csv" \
 --out="/home/anla/Source/python/style_feature_extraction/cluster/movies/gist/res3_" \
 --f_lab=1 --f_gist=0 --n_clusters=3

'''

