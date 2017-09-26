import os

def get_list_dir(path):
	return [os.path.join(path,name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def get_list_filename(path):
	return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]	

def get_list_filedir(path):
	return [os.path.join(path,name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]	
