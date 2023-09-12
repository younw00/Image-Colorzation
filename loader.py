from PIL import Image
import numpy as np
from os import  walk,PathLike
from os.path import join
from typing import Union

def load_img(img_path:Union[str,PathLike])->np.array:
	"""
	Loads in either grayscale or rgb images as RGB images. 
	If image is grayscale (only has one channel), the image is  repeated three times for the channels R,G, and B
	:param img_path: string or os.PathLike representing the path to the image 
	:returns 
	"""
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def image_loader(path, size):

	imgs = []
	counter = 0
	print(f"Loading images from {path}...")
	for _, _, files in walk(path):
		for name in files:
			if counter < size:
				img = load_img(join(path, name))
				imgs.append(img)
				counter += 1
	return imgs