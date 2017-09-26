import pandas as pd
import numpy as np
import urllib2
import re

label_list = None

def  download_img(url, savepath):
    img_data = None
    try:
        img_data = urllib2.urlopen(url).read()
    except urllib2.HTTPError, err:
        if err.code == 404:
            print "Page not found!"
        elif err.code == 403:
            print "Access denied!"
        else:
            print "Something happened! Error code", err.code
        return None
    except urllib2.URLError, err:
        print "Some other error happened:", err.reason  
        return None    
    
    img_path = '%s/%s' % (savepath, re.split('/', url)[-1]) 
    with open(img_path, 'wb') as handler:
        handler.write(img_data)
    return img_path

def get_label(row):
    for c in label_list:
        if row[c]==True:
            return c

def  download(img_urls, savepath):
    img_paths = [download_img(url, savepath) for url in img_urls]
    return img_paths

def  create_df(data_path, savepath):
    df = pd.read_csv(data_path)
    df.drop(df.columns[0], axis = 1, inplace = True)
    global label_list
    label_list = df.columns[3:-2]
    labels = df.apply(get_label, axis=1)
    img_paths = ['%s/%s' % (savepath, re.split('/', url)[-1]) for url in df['image_url']]
    df2 = pd.DataFrame({'img_path':img_paths, 'img_url': df['image_url'], 'label':labels, '_split':df['_split']})  
    df2['label'] = df2['label'].fillna('style_Vintage')
    train = df2[df2['_split'] == 'train']
    test = df2[df2['_split'] == 'test']

    del df2['_split']
    del train['_split']
    del test['_split']
    
    return df2, train, test

def main():
    data_path = '../../data/flickr/flickr_df_mar2014.csv'
    savepath = 'data/flickr/images'
    df, train, test = create_df(data_path, savepath)

    print df.head()    
    df.to_csv('files/flickrfull.csv')
    train.to_csv('files/flickr_train.csv')
    test.to_csv('files/flickr_test.csv')
    
#    for i in range(len(df)):
#        print (df['img_url'].iloc[i], '/home/anla/Source/example/flickr/images')
#        download_img(df['img_url'].iloc[i], '/home/anla/Source/example/flickr/images')

main()