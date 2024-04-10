import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import cv2
import random
from os import listdir

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)  
file1 = "./Example/train/raw"
mkdir(file1)  
file2 = "./Example/train/seg"
mkdir(file2)  
file3 = "./Example/ck"
mkdir(file3)  
file4 = "./Example/inference"
mkdir(file4)  
#%% Here we just have one training paires

path_raw = './training_paires/raw'
path_seg = './training_paires/seg'
num_train_paires_raw = listdir(path_raw)
num_train_paires_seg = listdir(path_seg)

im = io.imread(os.path.join(path_raw, num_train_paires_raw[0]))
label = io.imread(os.path.join(path_seg, num_train_paires_seg[0]))
plt.subplot(121)
plt.imshow(im);plt.title('Grayscale image')
plt.subplot(122)
plt.imshow(label);plt.title('Labeled image')

if len(label.shape) == 3:
    pixels = label.reshape(-1, 3)
    classes,_ = np.unique(pixels, axis=0, return_counts=True)
if len(label.shape) == 2:
    classes = np.unique(label)
print('Image contains: ', classes, "labels")

training_patch_size = 96
h, w = training_patch_size,training_patch_size
id = 0
#%% Random crop
for i in range(len(num_train_paires_raw)):
    im = io.imread(os.path.join(path_raw, num_train_paires_raw[i]))
    label = io.imread(os.path.join(path_seg, num_train_paires_seg[i]))
    count=0
    while 1: 
        y = random.randint(0, np.size(im,0)-training_patch_size)
        x = random.randint(0, np.size(im,1)-training_patch_size)
        gray = im[(y):(y + h), (x):(x + w)]
        lb = label[(y):(y + h), (x):(x + w)]
    
        cv2.imwrite(os.path.join(file1, str(id) + '.png'), gray)
        cv2.imwrite(os.path.join(file2, str(id) + '.png'), lb)
        count+=1
        id +=1
        if count == 800:
            break
print('Overall', id, 'patches has been generated for training')    
    