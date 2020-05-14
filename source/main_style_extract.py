from feature_extraction.cielab import cielab
from feature_extraction.lmgist import lmgist, param_gist

import cv2
import numpy as np
from sklearn.externals import joblib
from sklearn import svm

GIST_MAX = 0.430965412969
GIST_MIN = 0.000114381272033
# LAB_MAX = 25766
# LAB_MIN = 0

class   Feature_Parameter():
    def     __init__(self, lab, gist):
        self.lab = lab
        self.gist = gist
        self.gist_param = None
        if self.gist:
            self.gist_param = param_gist()
            self.gist_param.img_size = 256
            self.gist_param.orientations_per_scale = [8, 8, 8, 8]
            self.gist_param.number_blocks = 4
            self.gist_param.fc_prefilt = 4                    

class   Style_Feature(Feature_Parameter): 
    def     __init__(self, lab = 1, gist = 0, gist_processmodel = None):
        Feature_Parameter.__init__(self, lab, gist)
        self.gist_processmodel = None
        if gist == 1 and gist_processmodel is not None:
            print ('Load %s' % gist_processmodel)
            self.gist_processmodel = joblib.load(gist_processmodel)              
        
    def     get_feature(self, img):
        if img is None:
            return None
        if len(img.shape) < 3:
            return None
        feature = None
        if self.lab:
            feature = cielab(img)#(cielab(img) - LAB_MIN) / (LAB_MAX - LAB_MIN)
        if self.gist:
            img_tmp = cv2.resize(img, (256, 256))
            gist_feature = (lmgist(img_tmp, self.gist_param)[0].astype(float) - GIST_MIN) / (GIST_MAX - GIST_MIN)
            if self.gist_processmodel is not None:
                gist_feature = self.gist_processmodel.predict_proba(gist_feature)
            if feature is None:
               feature = gist_feature
            else:
               feature = np.concatenate((feature, gist_feature))
        return feature
        
        
DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
       
    img = np.array(mpimg.imread('../../example/1.jpg'))    
    style_obj = Style_Feature(lab = 1, gist = 0, gist_processmodel = '/home/anla/Source/python/style_feature_extraction/scence_recognition/models/full_gist_prob.pkl')
    feature = style_obj.get_feature(img)

    print (feature.shape, np.sum(feature))

    
#class   Style_Feature:
#    def    __init__(self, img):
#        self.img = None
#        if img is not None:
#            self.img = img
#        self.cielab = cielab(self.img)
#        
#        param = param_gist()
#        param.img_size = 256
#        param.orientations_per_scale = [8, 8, 8, 8]
#        param.number_blocks = 4
#        param.fc_prefilt = 4
#        self.gist, _ = lmgist(img, param)
#
#        self.feature = np.concatenate((self.cielab.astype(float), self.gist.astype(float)))

##========================================================##
#
#DEBUG = True
#
#if DEBUG:  
#    import matplotlib.pyplot as plt
#    import matplotlib.image as mpimg
#    
#    img = np.array(mpimg.imread('../../example/1.jpg'))
#    plt.imshow(img)
#    plt.show()
#    
#    style_object = Style_Feature(img)
#    print style_object.cielab.shape
#    print style_object.gist.shape
#    print style_object.feature.shape
## 
#import numpy as np
#from feature_extraction.cielab import cielab
#from feature_extraction.lmgist import lmgist, param_gist
#   
#def style_feature_extraction(img, lab, gist, gist_parameters = None):
#    if gist and gist_parameters is None:
#        param = param_gist()
#        param.img_size = 256
#        param.orientations_per_scale = [8, 8, 8, 8]
#        param.number_blocks = 4
#        param.fc_prefilt = 4        
#    feature = None
#
#    if lab == True:
#        feature = cielab(img).astype(float)
#    if gist == True:
#        if feature is None:
#            feature = lmgist(img, param)[0].astype(float)
#        else:
#            feature = np.concatenate((feature, lmgist(img, param)[0].astype(float)))
#    return feature
#
#
#DEBUG = False
#
#if DEBUG:  
#    import matplotlib.pyplot as plt
#    import matplotlib.image as mpimg
#    
#    img = np.array(mpimg.imread('../../example/1.jpg'))
#    style_feature = style_feature_extraction(img)
#    print style_feature