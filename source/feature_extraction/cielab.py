# -*- coding: utf-8 -*-
import cv2
import numpy as np

def     cielab(img):
    '''
        Calculate CIELAB feature from input image
    '''
    #check if image exist
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # print (labimg)
    # print (np.max(labimg[0, :, :]), np.min(labimg[0, :, :]))
    # print (np.max(labimg[:, 0, :]), np.min(labimg[:, 0, :]))
    # print (np.max(labimg[:, :, 0]), np.min(labimg[:, :, 0]))
    channels = [0, 1, 2]
    histsize = [4, 14, 14]
    histrange = [0, 256, 0, 256, 0, 256]

    hist = cv2.calcHist(images = cv2.split(labimg), channels = channels, mask = None, histSize = histsize, ranges = histrange)

    # print (np.array(hist).astype(int))
    hist = hist.flatten()
    sumhist = np.sum(hist)
    if sumhist == 0:
        return None
    hist /= sumhist
    return hist
    
##========================================================##

DEBUG = False

if DEBUG:  
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    img = np.array(mpimg.imread('/mnt/e/Wallpaper/Phoenix.jpg'))
#    img = np.array(mpimg.imread('/home/anla/Source/example/flickr/images/5159730964_124c1e63d8.jpg'))
    plt.imshow(img)
    plt.show()
    
    LABhist = cielab(img)
    if LABhist is not None:
        print (LABhist)
        
        print (LABhist.shape)
        print (img.shape, np.sum(LABhist))
