# Style Extraction for Images

Extract feature GIST and LAB from images, then experiments with following tasks:

+ Scence Recognition: on [Torralba dataset](http://people.csail.mit.edu/torralba/code/spatialenvelope/) about outdoor images (forest, mountain...)
+ Style Recognition: on Flickr dataset about styles of book covers (vintage, romantic...).
+ Clustering images

Using data of movie service, extract GIST and LAB from posters to represent movies. Considering users watch these movies:

+ Clustering users
+ Recommending movies which are similar in genres to history of users

## 1. Feature extraction

- CIELAB (Lab color histogram: 4|14|14 bins in L|a|b channel)
- GIST
  + http://people.csail.mit.edu/torralba/code/spatialenvelope/
  + https://www.quora.com/Computer-Vision-What-is-a-GIST-descriptor
  + https://prateekvjoshi.com/2014/04/26/understanding-gabor-filters/

Scripts are provided at source/feature_extraction

## 2. Data preparation

Scripts dowloading data and pre-processing data are provided at source/data_preprocessing.		  

## 3. Experiences

### 3.1 Scence Recognition

- Data: 2688 images from [here](http://people.csail.mit.edu/torralba/code/spatialenvelope/)
- Accuaracy:
    + Torralba's work: 83%
    + LAB:      Train: 51.86%, Test: 41.82%
    + GIST:     Train: 92.88%, Test: 86.25%
    + GIST-LAB: Train: 95.12%, Test: 88.48% (feature fusion by SVM)

Check scripts at [source/classification](https://github.com/anvy1102/style-etraction-for-images/tree/master/source/classification) and [source/main_classify.py](https://github.com/anvy1102/style-etraction-for-images/tree/master/source/main_classify.py)     

### 3.2 Style Recognition

- Flick dataset
	+ LAB:      Train: 8.58%, Test: _
	+ LAB:      Train: 22.6%, Test: _

Running code is similar to 3.1

### 3.3 Clustering

+ Clustering by Gist feature: 

![gist_clustering_image](/images/gist_clustering_image.png)

+ Clustering by Gist+Lab feature: 
	
![gistlab_clustering_image](/images/gistlab_clustering_image.png)

+ Clustering users by GMM

Check scripts at [source/clustering](https://github.com/anvy1102/style-etraction-for-images/tree/master/source/clustering) and [source/main_cluster.py](https://github.com/anvy1102/style-etraction-for-images/tree/master/source/main_cluster.py)

### 3.4 Recommending

Check scripts at [source/recommending](https://github.com/anvy1102/style-etraction-for-images/tree/master/source/recommending) and [source/main_recommend.py](https://github.com/anvy1102/style-etraction-for-images/tree/master/source/main_recommend.py)     
## 4. Dependencies 

- Anaconda4.4 - Python 2.7 (numpy, pandas, matplotlib)
- Pytorch (conda install pytorch torchvision -c soumith)
- opencv-3.1.0 (conda install opencv)
- sklearn


## Ref	
- [Recognizing Image Style](https://arxiv.org/abs/1311.3715)

- [Methods for merging Gaussian mixture components](https://doi.org/10.1007/s11634-010-0058-3)

- [Combining Mixture Components
for Clustering](https://www.stat.washington.edu/raftery/Research/PDF/Baudry2010.pdf)





