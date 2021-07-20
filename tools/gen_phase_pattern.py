# python temp/gen_phase_pattern.py
import os
import numpy as np
import cv2
import argparse


width, height = 1280, 800
num_of_period = 64
start_id = 25 # 1


print("num_of_period: " + str(num_of_period))

def generate_gray_pattern():
    num_of_period_for_gray = num_of_period * 2
    pix_per_period = width // num_of_period_for_gray
    num = len(bin(num_of_period_for_gray-1))-2
    print("gray oce pattern num (+1 for edge checking):" + str(num))
    print("pix_per_period for graycode:" + str(pix_per_period))
    # pix_id = x // pix_per_period
    imgs_code = 255*np.fromfunction(lambda y,x,n: (((x // pix_per_period)^((x // pix_per_period)>>1))&(1<<(num-1-n))!=0), (height,width,num), dtype=int).astype(np.uint8)
    imlist = list(imgs_code.transpose(2, 0, 1))
    return imlist

def generate_phase_shifting_pattern():
    print("pix_per_period for phase_shifting:" + str(width//num_of_period))
    num, F = 4, num_of_period

    w = 2*np.pi/width * F

    imgs_code = (255*np.fromfunction(lambda y,x,n: 0.5*(np.cos(w*x + 2*np.pi*n/num) + 1), (height,width,num), dtype=float)).astype(np.uint8)
    
    imlist = list(imgs_code.transpose(2, 0, 1))
    return imlist

def get_id_str(imgid):
    if imgid < 10: id_str = '0' + str(imgid)
    else: id_str = str(imgid)
    return id_str
    


imlist = generate_gray_pattern()

img_open = 255*np.ones_like(imlist[0]).astype(np.uint8)
img_close = np.zeros_like(imlist[0]).astype(np.uint8)
cv2.imwrite('./pattern'+get_id_str(start_id)+'.bmp', img_open)
start_id += 1
cv2.imwrite('./pattern'+get_id_str(start_id)+'.bmp', img_close)
start_id += 1

for i in range(len(imlist)):
    cv2.imwrite('./pattern'+get_id_str(i+start_id)+'.bmp', imlist[i])


start_id += len(imlist)
start_id = 39
imlist = generate_phase_shifting_pattern()
for i in range(len(imlist)):
    cv2.imwrite('./pattern'+get_id_str(i+start_id)+'.bmp', imlist[i])
exit()
