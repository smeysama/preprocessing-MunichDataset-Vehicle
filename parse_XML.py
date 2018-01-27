
# Author: Seyed Majid Azimi
# Date: 24 Jan 2018

import xmltodict
import numpy as np
import pandas as pd
from skimage.transform import rotate
import matplotlib


#matplotlib.use('Gtkagg')
#matplotlib.use('Tkagg')
#matplotlib.use('agg')

import matplotlib.pyplot as plt

import os
import re
import matplotlib.image as mpimage
import math
import numpy.matlib

# rotate original images by 90,180, 270 and flip them horizontally and vertically and modify the annotation accordingly and visualize the annotations



# def to_dict(path):
#     l = ''
#     for line in open(path):
#         l += line
#
#     d = xmltodict.parse(l)
#     image = {'name': d['annotation']['source']['image'], 'objects': {
#         i: {
#
#         }
#
#         for i, car in enumerate(d['annotation']['object'])
#         }}
#
#     for i, car in enumerate(d['annotation']['object']):
#         image['objects'][i]['name'] = car['name']
#         for key in car['bndbox']:
#             image['objects'][i][key] = car['bndbox'][key]
#
#     # return image
# def r(x, y, angle, center):
#     x_, y_ = [], []
#     for i in range(len(x)):
#         newX = center[0] + (x[i] - center[0]) * np.cos(angle * np.pi / 180) - (y[i] - center[1]) * np.sin(
#             angle * np.pi / 180)
#         newY = center[1] + (x[i] - center[0]) * np.sin(angle * np.pi / 180) + (y[i] - center[1]) * np.cos(
#             angle * np.pi / 180)
#         x_.append(newX)
#         y_.append(newY)
#     return x_, y_
#
#
# def rotate_crop(image, angle, center, size, i):
#     c = r([center[0]], [center[1]], angle, [len(image[0]) / 2, len(image[0]) / 2])
#     new_center = (c[0][0], c[1][0])
#     new_image = rotate(image,-angle)
#     return new_image

#MAIN_DIR='/home/azim_se/MyProjects/ECCV18/data/DLR/Images'
#addtop

BASE_DIR = './Train/'
CROP_DIR = './Train/'

VERBOSE = False
VERBOSE_points = False

label = {10:'pkw', 11:'pkw_trail', 16:'van', 17:'van_trail', 22:'truck', 23:'truck_trail', 20:'cam', 30:'bus', 12:'motorcycle'}
#def rotateBig(img, angle=0):
#    image = rotate(img, angle)
#    return rotimage

def code(annotation):
    # print('code : ' ,re.findall('4K0G[0-9]+', annotation)[0])
    return re.findall('4K0G[0-9]+', annotation)[0]


def info(path):
    com_path = BASE_DIR + path
    file = open(com_path)
    cars = []
    for line in file:
        if not (line[0] == '#' or line[0] == '@'):
            line = list(map(float, line.split()[0:]))
            # degree is enough to be integer, no need to be float.
            cars.append({
		'id' :line[0],
                'type': int(line[1]),
                'center.x': line[2],
                'center.y': line[3],
                'size_width': line[4],
                'size_height': line[5],
                'angle': int(line[6])
            })
    if VERBOSE:
        print(cars)
    return cars


def parse_file(images_name, annotation):
    images_annotation = {
        image_name: [info(path) for path in annotation if code(path) in image_name]
        for image_name in images_name
        }
    if VERBOSE:
        print(images_annotation.keys())
        
    for key in images_annotation.keys():
        l = []
        for i in range(len(images_annotation[key])):
            for j in range(len(images_annotation[key][i])):
                if VERBOSE:
                        print(images_annotation[key][i][j])
                l.append(images_annotation[key][i][j])
        images_annotation[key] = l
        if VERBOSE:
            print(images_annotation[key])
    return images_annotation


def dict_to_data_frame(d):
    if VERBOSE:
        print(len(d))
        print(len(d[0].keys()))
        
    df = pd.DataFrame(data=np.zeros((len(d), len(d[0].keys()))), columns=d[0].keys())
    for i, car in enumerate(d):
        if VERBOSE:
            print(i)
            print(car)
            
        for col in df.columns:
            if VERBOSE:
                print('Each columns of DataFormat of Panda: {}', format(d[i][col]))
            df.set_value(i, col, d[i][col])
            if VERBOSE:
                print(d[i][col])
    if VERBOSE:
        print('DataFormat Panda whole samples of each image:')
        print(df)
        
    return df


def rotate_(x, y, angle, center):
    x_, y_ = [], []
    for i in range(len(x)):
        newX = center[0] + (x[i] - center[0]) * np.cos(angle * np.pi / 180) - (y[i] - center[1]) * np.sin(
            angle * np.pi / 180)
        newY = center[1] + (x[i] - center[0]) * np.sin(angle * np.pi / 180) + (y[i] - center[1]) * np.cos(
            angle * np.pi / 180)
        x_.append(newX)
        y_.append(newY)
    return x_, y_


def magic(image, images_cars, kernel_size, r_stride, c_stride):
    row = 0
    image_dim_0, image_dim_1 = np.shape(image)[0], np.shape(image)[1]
    cropped = []
    while row + kernel_size < image_dim_0:
        col = 0
        while col + kernel_size < image_dim_1:

            cars_in_this_sub_image = []

            for car in range(len(images_cars)):
                if col <= images_cars.iloc[car]['center.x'] <= col + kernel_size and row <= images_cars.iloc[car]['center.y'] <= row + kernel_size:

                    cars_in_this_sub_image.append({
			'id' : images_cars.iloc[car]['id'],
                        'type': int(images_cars.iloc[car]['type']),
                        'center.x': images_cars.iloc[car]['center.x'] - col,
                        'center.y': images_cars.iloc[car]['center.y'] - row,
                        'size_width': images_cars.iloc[car]['size_width'],
                        'size_height': images_cars.iloc[car]['size_height'],
                        'angle': images_cars.iloc[car]['angle']
                    })
            cropped.append((image[row:row + kernel_size, col:col + kernel_size, :], cars_in_this_sub_image))
            col += c_stride
        row += r_stride
    return cropped


def plot(image, annotation):
    b=np.zeros((2,1))
    ax = plt.imshow(image)
    Id = annotation['id'].values
    x, y = annotation['center.x'].values, annotation['center.y'].values
    w, h = annotation['size_width'].values, annotation['size_height'].values
    a = annotation['angle'].values
    t = annotation['type'].values
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }
    if VERBOSE_points:
        print(x[0])
        print(y)
        print(w)
        print(h)
        print(a)
    print(t)
    
    for i in range(len(x)):
            b[0,0]=x[i]
            b[1,0]=y[i]
            t_ = t[i]
            if t_ in [22,23,20,30]:
                extra_w = 25
                extra_h = 0
            else:
                extra_w = 15
                extra_h = 5

            w_ = w[i] + extra_w
            h_ = h[i] + extra_h
            a_ = a[i]
            id_ = Id[i]

            bbox = np.matlib.repmat(b,1,5)+np.matmul([[math.cos(math.radians(a_)), math.sin(math.radians(a_))],[-math.sin(math.radians(a_)), math.cos(math.radians(a_))]], [[-w_/2, w_/2, w_/2, -w_/2, w_/2+8], [-h_/2, -h_/2, h_/2, h_/2, 0]])
	
            plt.plot([bbox[0][0],bbox[0][1]],[bbox[1][0],bbox[1][1]], linewidth=1, color='magenta')
            ###plt.plot([bbox[0][1],bbox[0][2]],[bbox[1][1],bbox[1][2]])
            plt.plot([bbox[0][2],bbox[0][3]],[bbox[1][2],bbox[1][3]],linewidth=1, color='magenta')
            plt.plot([bbox[0][0],bbox[0][3]],[bbox[1][0],bbox[1][3]],linewidth=1, color='magenta')
            
            plt.plot([bbox[0][1],bbox[0][4]],[bbox[1][1],bbox[1][4]], linewidth=1, color='cyan')
            plt.plot([bbox[0][2],bbox[0][4]],[bbox[1][2],bbox[1][4]], linewidth=1, color='cyan')
            
            #plt.plot([bbox[0][1],bbox[0][4]],[bbox[1][1],bbox[1][4]], linewidth=3.5, color='red', marker='*', linestyle='dashed', markersize=5, markeredgecolor='red')
            #plt.plot([bbox[0][2],bbox[0][4]],[bbox[1][2],bbox[1][4]], linewidth=3.5, color='red', marker='*', linestyle='dashed', markersize=5, markeredgecolor='red')
            #plt.text(int(min(bbox[0,:])), int(min(bbox[1,:])) - 2, str(int(t_)), fontdict=font)

            plt.text(int(min(bbox[0,:])), int(min(bbox[1,:])) - 2,
                    '{:s}, {:s}, {:s}'.format(str(id_),label[t_],str(a_)),
                    bbox=dict(facecolor='blue', alpha=0),
                   fontsize=10, color='red')
    plt.scatter(x, y)
    plt.show()


images_name = [name for name in os.listdir('./Train/') if 'JPG' in name]
annotation = [name for name in os.listdir('./Train/') if 'samp' in name]

if VERBOSE:
    print(images_name)
    print(len(annotation))
    
f = parse_file(images_name, annotation)
df = dict_to_data_frame(f[images_name[0]])


a = BASE_DIR+images_name[0]
print(a)
image = mpimage.imread(a)
#plt.imshow(image)
#plt.show()

plot(image, df)

'''
cropped = magic(image, df, kernel_size=960, r_stride=100, c_stride=100)
for sub in cropped :
    if len(sub[1])>0:
        df  = dict_to_data_frame(sub[1])
        plot(sub[0],df)
        _270 = rotate(sub[0],270)
        _df = df
        _df['center.x'] , _df['center.y'] =  rotate_(df['center.x'].values,df['center.y'].values,-270, [960 / 2,960 / 2])
        plot(_270,_df)
'''
