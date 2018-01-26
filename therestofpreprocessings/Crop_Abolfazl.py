#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Purpose: 
Input: 
Output:
Usage: Python3 yolo.py
Author: Abolfazl Meyarian
Date:
Organization:
"""

# add packages
import numpy as np
import pandas as pd
import itertools
import math
import matplotlib.pyplot as plt
from termcolor import colored
import os
from shapely.geometry import box
from shapely.affinity import rotate as r
import matplotlib.image as mpimage
import re
from skimage.transform import rotate
from bs4 import BeautifulSoup

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def crop_main(image, kernel_size, r_stride, c_stride):
	image_dim_0, image_dim_1 = np.shape(image)[0], np.shape(image)[1]
	print('image_dim_0, image_dim_1 are {} & {}'.format(image_dim_0, image_dim_1))
	cropped = []
	A = [i for i in range(1,image_dim_0,r_stride)]
	B = [i for i in range(1,image_dim_1,c_stride)]
	hh = [ A , image_dim_0 - r_stride - 1] 
	ww = [B ,  image_dim_1 - c_stride - 1]
	print(hh)
	print(ww)    
	for h in hh:
        	for w in ww:
            		cropped.append(image[h:min(h+r_stride-1,h1),w:min(w+c_stride-1,w1),:])

	return cropped




def magic(image, kernel_size, r_stride, c_stride):
	row = 0
	image_dim_0, image_dim_1 = np.shape(image)[0], np.shape(image)[1]
	print('image_dim_0, image_dim_1 are {} & {}'.format(image_dim_0, image_dim_1))
	cropped = []
	while row + kernel_size < image_dim_0:
		col = 0
		while col + kernel_size < image_dim_1:
			cropped.append(image[row:row + kernel_size, col:col + kernel_size, :])
			col += c_stride
		cropped.append(image[row:row + kernel_size, image_dim_1 - kernel_size:, :])
		print('col and row are {} & {}'.format(col, row))
		row += r_stride
		col = 0
		if  row + kernel_size > image_dim_0:
			while col + kernel_size < image_dim_1:
				cropped.append(image[image_dim_0 - kernel_size:, col:col + kernel_size, :])
				col += c_stride
			row = 0
	return cropped

'''
def sample_noise(shape):
    return tf.random_uniform(shape=shape, minval=-1, maxval=1)


def add_noise(input, shape):
    return input + sample_noise(shape)


def read_image(address, format):
    print('reading from : ', colored(address, 'blue'))
    image_dir = [address + name for name in os.listdir(address) if format in name.lower()]
    print(image_dir)
    images = [plt.imread(name) for name in image_dir]
    print(colored('reading finished.', 'green'))
    return np.array(images)
'''

# __________________________________________________________________
# __________________________________________________________________
# __________________________________________________________________


image_dir = '../cropped_image/'
image_dim = 960


show_result = True
_mode=['train','test']
BASE_DIR = './data/'
CROP_DIR = './cropped/'
_data=['image','annotation']
kernel_size=960
r_stride={'train':100,'test':kernel_size-30}
c_stride={'train':100,'test':kernel_size-30}
name_counter = 0

for mode in _mode:
	for data in _data:
		print ("mode and data are {} and {}".format(mode, data))
		if data=='image':
			images_name = [name for name in os.listdir(BASE_DIR + data + '/' + mode + '/') if 'jpg' in name]
		else:
			images_name = [name for name in os.listdir(BASE_DIR + data + '/' + mode + '/') if 'png' in name]


		for name in images_name:

			print(colored('reading image:' + name, 'green'))
			image = mpimage.imread(BASE_DIR + data + '/' + mode + '/' + name)
			print('done.')

			print(colored('cropping started...', 'green'))
			cropped = crop_main(image, kernel_size, r_stride[mode], c_stride[mode])
			print('done.')
			if _mode=='train':
				print("mode train, with augmentation!")
				for image in cropped:
					_90 = rotate(image, 90)
					_180 = rotate(image, 180)
					_270 = rotate(image, 270)
					if data=='image':
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + '90.jpg', arr=_90)
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + '180.jpg', arr=_180)
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + '270.jpg', arr=_270)
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + 'main.jpg', arr=image)
					elif data=='annotation':
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + '90.png', arr=_90)
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + '180.png', arr=_180)
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + '270.png', arr=_270)
		        			mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + 'main.png', arr=image)

			elif _mode=='test':
				print("mode test, no augmentation!")
				if data=='image':
		        		mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + 'main.jpg', arr=image)
				elif data=='annotation':
		        		mpimage.imsave(CROP_DIR + data + '/' + mode + '/' + str(name_counter) + '_' + 'main.png', arr=image)

print(colored(name, 'green'), 'done.')
print(colored('----------------------', 'blue'))

