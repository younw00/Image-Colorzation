from skimage import color
import torch
import cv2 
from typing import Tuple
import numpy as np 
from PIL import Image
import numpy as np
import torch.nn.functional as F

def L_to_tensor(img_rgb_orig):
	# return L layer as torch Tensor

	img_lab_orig = color.rgb2lab(img_rgb_orig)

	img_l_orig = img_lab_orig[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,:,:]

	# print(tens_orig_l.shape)

	return tens_orig_l

def AB_to_tensor(img_rgb_orig):
	# return L layer as torch Tensors
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)

	img_A_orig = img_lab_orig[:,:,1]
	img_B_orig = img_lab_orig[:,:,2]

	tens_orig_A = torch.Tensor(img_A_orig)[None,:,:]
	tens_orig_B = torch.Tensor(img_B_orig)[None,:,:]
	tens_orig_AB = torch.cat((tens_orig_A, tens_orig_B), 0)

	# print(tens_orig_AB.shape)

	# tens_orig_AB = torch.Tensor(img_AB_orig)[None,:,:]
	# tens_orig_AB = torch.Tensor(img_AB_orig)[None,2,:,:]

	return tens_orig_AB

def imgs_tensor(imgs): # this is literally not a tensor ??? 
	batch_label = []
	for img in imgs:
		batch = L_to_tensor(img)
		label = AB_to_tensor(img)
		batch_label.append((batch, label))
	return batch_label

def resize_to_shape(imgs,targ_shape:Tuple[int,int]):
	result_imgs = []
	for img in imgs: 
		result_imgs.append(cv2.resize(img,targ_shape))
	return imgs 

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def img_to_bins(images):
	# images is a list of tensors or a tensor as shown:  
	#    tensor:  a tensor of shape N x [2,3] x H x W 
	# 	 list of tensors: a list of tensors of shape 1 x x [2,3] x H x W 
	
	images_mat = torch.stack([AB_to_tensor(img) for img in images]) # regularize to single tensor rather than list of tensors 
	ims = np.floor_divide(images_mat,10)
	if ims.shape[1] == 3:
		ims = ims[:,1:3,:,:]
	assert(ims.shape[1] == 2)
	print(ims.shape)
	ims = np.swapaxes(ims,1,-1)
	print(ims.shape)
	ims = np.reshape(ims,-1,ims.shape[-1])
	print(ims.shape)


def preprocess_img(img_rgb_orig, HW=(64,64), resample=3):
  # return original size L and resized L as torch Tensors
  img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
  
  img_lab_orig = color.rgb2lab(img_rgb_orig)
  img_lab_rs = color.rgb2lab(img_rgb_rs)

  img_l_orig = img_lab_orig[:,:,0]
  img_l_rs = img_lab_rs[:,:,0]

  img_ab = img_lab_rs[:,:,1:3]

  tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
  tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

  return tens_orig_l, tens_rs_l,img_ab

def postprocess(tens_orig_l, out_ab,gamma=1.0, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	images =  color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

	if gamma != 1.0: 
		print('interpolating')
		# images = np.power(images,gamma)
	return images 
