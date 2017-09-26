import cv2
import numpy as np

def load_img(img_path):
    return np.array(cv2.imread(img_path))


#def get_list_img(path_data):
#	list_videos = get_list_dir(path_data)
#	list_imgs = [[load_img(img_path) for img_path in video] for video in list_videos]
#	return list_imgs




