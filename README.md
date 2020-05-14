# Style Extraction for Images
Extract style from images

## Dependencies 

- Anaconda4.4 - Python 2.7 (numpy, pandas, matplotlib)
- Pytorch (conda install pytorch torchvision -c soumith)
- opencv-3.1.0 (conda install opencv)

## Features
- CIELAB (Lab color histogram: 4|14|14 bins in L|a|b channel)
- GIST
  + http://people.csail.mit.edu/torralba/code/spatialenvelope/
  + https://www.quora.com/Computer-Vision-What-is-a-GIST-descriptor
  + https://prateekvjoshi.com/2014/04/26/understanding-gabor-filters/

## Experiences
### Scence Recognition
    - Data: 2688 images from [here](http://people.csail.mit.edu/torralba/code/spatialenvelope/)
    - Accuaracy:
        + Torralba's work: 83%.
        + GIST: 86.25%
        + GIST-LAB: 88.48%
        
### Image Clustering



https://github.com/anvy1102/style-etraction-for-images/blob/master/source/clustering/.ipynb_checkpoints/Clustering-checkpoint.ipynb




