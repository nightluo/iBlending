# Packages
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from torchvision.utils import save_image
from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, MeanShift, Vgg16, gram_matrix
import argparse
import pdb
import os
import imageio.v2 as iio
import torch.nn.functional as F
import utils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--source_root', type=str, default='/mnt/data/luoyan/road/ib/source/', help='path to the source image')
parser.add_argument('--mask_root', type=str, default='/mnt/data/luoyan/road/ib/masks/', help='path to the mask image')
parser.add_argument('--img_name', type=str, default='1.png', help='')

parser.add_argument('--output_dir', type=str, default='./results/masks/', help='path to output')

parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

opt = parser.parse_args()

os.makedirs(opt.output_dir, exist_ok = True)
os.makedirs(opt.mask_root, exist_ok = True)

root = opt.source_root
for i in os.listdir(root):
    print(i)
    path = os.path.join(root, i)
    L_img = np.array(Image.open(path).convert('L'))
    img = np.array(Image.open(path))
    w = img.shape[0]
    h = img.shape[1]
    for x in range(w):
        for y in range(h):
            r, g, b, a = img[x ,y]
            if a == 0:
                L_img[x, y] = 0
            else:
                L_img[x, y] = 255
    
    imsave(os.path.join(opt.mask_root, i), L_img.astype(np.uint8))
    


# img_name = opt.img_name
# source_file = os.path.join(opt.source_root, opt.img_name)

# # source_img = np.array(Image.open(source_file).convert('RGB'))
# # imsave(os.path.join(opt.output_dir, 'source_img_RGB.png'), source_img.astype(np.uint8))
# # print(f"source_img:{source_img}")

# L_img = np.array(Image.open(source_file).convert('L'))


# img = np.array(Image.open(source_file))
# imsave(os.path.join(opt.output_dir, 'source_img_RGBA.png'), img.astype(np.uint8))
# print(f"source_img:{img}")
# w = img.shape[0]
# h = img.shape[1]
# for x in range(w):
#     for y in range(h):
#         r, g, b, a = img[x ,y]
#         if a == 0:
#             L_img[x, y] = 0
#         else:
#             L_img[x, y] = 255

# imsave(os.path.join(opt.output_dir, 'L_img.png'), L_img.astype(np.uint8))

# ret, thresh = cv2.threshold(source_img, 200, 255,cv2.THRESH_BINARY)
# imsave(os.path.join(opt.output_dir, 'source_img_150.png'), thresh.astype(np.uint8))
# print(f"source_img:{source_img}")

