import torch
import torchvision.transforms as tfs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import get_efficientunet_b5
from model import Uresnet
from tqdm import tqdm
import os
import argparse
import nibabel as nib
from skimage import io
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UresNet', 
                    help='define which gpu yo use, can be "UresNet" or "efficientUnet"')
parser.add_argument('--cpu_gpu', type=str, default='gpu', 
                    help='define which gpu yo use, can be "gpu" or "cpu"')
parser.add_argument('--single_fulldomain', type=str, default='Full', 
                    help='define which gpu yo use')
parser.add_argument('--input', type=str, default='3D', 
                    help='define your input size, can be "2D" or "3D"')

#%% This size define is very important, for now please use nteger multiples of the training patch size
parser.add_argument('--size', type=int, default=768, help='Load checkpoint')
#===============================================================================================
parser.add_argument('--checkpoint', type=int, default=6, help='Load checkpoint')
parser.add_argument('--gpu', type=int, default=0, help='define which gpu yo use')
parser.add_argument('--dir_3D_data', type=str, default='H:\\Segment_everthing\\Example\\test\\3D',
                    help='3D image directory')
parser.add_argument('--dir_2D_data', type=str, default='H:\\Segment_everthing\\Example\\test\\2D',
                    help='2D testing image directory')
parser.add_argument('--dir_checkpoint', type=str, default='H:\\Segment_everthing\\Example\\ck',
                    help='Trained model directory')
parser.add_argument('--dir_result', type=str, default='H:\\Segment_everthing\\Example\\inference',
                    help='Segmented result directory')

opt = parser.parse_args()
print(opt)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)   
        
if opt.model == 'UresNet':
    net = Uresnet(3,3)   
    print('============== UresNet is used ===================')
elif opt.model == 'efficientUnet':
    net = get_efficientunet_b5(3) 
    print('============== EfficientUnet is used ===================')

checkpoint = torch.load(os.path.join(opt.dir_checkpoint, '{}.pt'.format(opt.checkpoint)))
net.load_state_dict(checkpoint['net_state_dict'])
if opt.cpu_gpu == 'gpu':
    net.cuda()
    print('============== GPU is used ===================')
    
net.eval()
im_tfs = tfs.Compose([
      tfs.ToTensor(),
      ])    

def crop(data,height=opt.size, width=opt.size):
    st_x = 40
    st_y = 40
    box = (st_x, st_y, st_x+width, st_y+height)
    data = data.crop(box)
    return data

if opt.input == "3D":
    name = listdir(opt.dir_3D_data)
    im = nib.load(os.path.join(opt.dir_3D_data, name[0]))
    img = im.get_data()
    
    file1 = os.path.join(opt.dir_result,'raw')
    mkdir(file1)    
    file2 = os.path.join(opt.dir_result,'seg')
    mkdir(file2)   
    
    for i in tqdm(range(0,np.size(img,2))):
        img1 = img[:,:,i]
        img2 = Image.fromarray(img1)
        img2 = crop(img2)
        cut_image = img2.convert('RGB')
        cut_image1 = im_tfs(cut_image)
        
        test_image1 = cut_image1.unsqueeze(0).float()
        with torch.no_grad(): 
            out = net(test_image1.cuda())
        pred = out.max(1)[1].squeeze().data.cpu().numpy()
        pred = np.uint8(pred)
        pred = Image.fromarray(pred)
        pred.save(file2 + '\\' + '%d' % i + '.png')
        cut_image.save(file1 + '\\' +  '%d' % i + '.png')
        
elif opt.input == "2D":
    name = listdir(opt.dir_2D_data)
    im = io.imread(os.path.join(opt.dir_2D_data, name[0]))

    img2 = Image.fromarray(im)
    img2 = crop(img2)
    cut_image = img2.convert('RGB')
    
    plt.subplot(121)
    plt.imshow(cut_image)
    
    cut_image1 = im_tfs(cut_image)
    test_image1 = cut_image1.unsqueeze(0).float()
    with torch.no_grad(): 
        out = net(test_image1.cuda())
    pred = out.max(1)[1].squeeze().data.cpu().numpy()
    pred = np.uint8(pred)
    pred = Image.fromarray(pred)
    plt.subplot(122)
    plt.imshow(pred)
    
    
    
    
    
    
    
    
    
    
    
    