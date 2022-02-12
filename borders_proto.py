from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
# import cv2
#
# imageA = cv2.imread("test_images/frame46243")
# imageB = cv2.imread("test_images/frame46244")
# # convert the images to grayscale
# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
import cv2 as cv
import os
import time
import numpy as np
import argparse
import random as rng
rng.seed(12345)
from scipy.ndimage.interpolation import shift
from video_proto import structural_similarity, compare_images

def nice_print(mas):
    s = [[str(e) for e in row] for row in mas]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    return


def compate_images_MSE (img1, img2):
    return(np.square(np.subtract(img1,img2)).mean())

def border_check(file_1, file_2):
    img_1 = cv.imread(file_1)
    img_2 = cv.imread(file_2)
    src_gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    src_blur_1 = cv.blur(src_gray_1, (3, 3))
    src_gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    src_blur_2 = cv.blur(src_gray_2, (3, 3))
    canny_1 = cv.Canny(src_blur_1, 100, 200)
    canny_2 = cv.Canny(src_blur_2, 100, 200)

    ssim_min = 1
    ssim = 1
    crop_size = 5
    for i in range(-5,5):
        for j in range(-5,5):
            frame_1 = shift(canny_1,[i,j])[crop_size:len(canny_2)-crop_size,crop_size:len(canny_2)-crop_size]
            frame_2 = canny_2[crop_size:len(canny_2)-crop_size,crop_size:len(canny_2)-crop_size]
            ssim = compare_images(frame_1, frame_2)
            if ssim<ssim_min:
                ssim_min=ssim
    return(ssim)



files = sorted(os.listdir('data/test/'))
prev = "data/test/frame45879.bmp"
for file in files:
    if not '.bmp' in file:
        continue
    current="data/test/{0}".format(file)
    print(current, border_check(current,prev))
    prev = current


