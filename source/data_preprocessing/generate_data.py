import numpy as np
from random import shuffle
import re 
import pandas as pd
from datapath_process import get_list_dir, get_list_filename

def generate_data(data_folder):
	print 'Reading ...'
	folders = get_list_dir(data_folder)
	def  get_type_img(img_name):
		if img_name.startswith("Tap_"):
			return "series"
		if img_name.startswith("standing__"):
			return "standing"
		if img_name.startswith("small__"):
			return "small"
		if img_name.startswith("big__"):
			return "big"
		return 'unknown'

	data_df = pd.DataFrame(columns=['img_name', 'video_id', 'img_type', 'img_path'])

	print "Number of folder: ", len(folders)
	for i in range(len(folders)):
		folder = folders[i]
		if i and i % 1000 == 0:
			print 'Processed %d folders' % (i)
		img_names = get_list_filename(folder)
		img_paths = ['%s/%s' % (folder, img) for img in img_names]
		video_ids = [re.split('/', folder)[-1]] * len(img_names)
		img_types = [get_type_img(img) for img in img_names]
		df_tmp = pd.DataFrame({'img_name':img_names, 'video_id': video_ids, 'img_type':img_types, 'img_path': img_paths})    
		data_df = data_df.append(df_tmp)	

	data_df.to_csv('images_relativepath.csv')
	print data_df.head()
	
data_folder = '../../data/images/videos/'	
generate_data(data_folder)