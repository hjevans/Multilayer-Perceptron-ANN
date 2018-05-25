# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:35:52 2017

@author: Hannah
"""

import cv2
import matplotlib.pyplot as plt

im = plt.imread('roombas/Roomba01.jpg')
plt.imshow(im)
x0 = 320
y0 = 240
plt.plot(x0,y0,'*m')
for i in range (360):
    M = cv2.getRotationMatrix2D((x0,y0),i,1)
    rot = cv2.warpAffine(im,M,(im.shape[1],im.shape[0]))
    a = rot[y0-100:y0+100,x0-100:x0+100,:]
    plt.figure(2)
    plt.imshow(rot)
    plt.plot(x0,y0,'*c')
    plt.figure(3)
    plt.imshow(a)
    cv2.imwrite('roombas/train/roomba01_%d.jpg' % i,a)

img = plt.imread('roombas/Roomba02.jpg')  
x1 = 255
y1 = 180
plt.figure(4)
plt.imshow(img)
plt.plot(x1,y1,'*m')
for j in range (360):
    M1 = cv2.getRotationMatrix2D((x1,y1),j,1)
    rot1 = cv2.warpAffine(img,M1,(img.shape[1],img.shape[0]))
    b = rot1[y1-100:y1+100,x1-100:x1+100,:]
    plt.figure(5)
    plt.imshow(rot1)
    plt.plot(x1,y1,'*c')
    plt.figure(6)
    plt.imshow(b)
    cv2.imwrite('roombas/train/roomba02_%d.jpg' % j,b)
    
    