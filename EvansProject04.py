# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:40:33 2017

@author: Hannah
"""

import pickle
import glob
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor as MLP

with open('ANN.pkl','rb') as a:
    ANN = pickle.load(a)
with open('scale.pkl','rb') as b:
    scale = pickle.load(b)

fname = glob.glob('roombas/02/*.jpg')
trueAngle = []
im = []
gray = []
dataHSV = []
HOG = [0]*len(fname)
histH = [0]*len(fname)
histS = [0]*len(fname)

for i in range(len(fname)):
    beg = fname[i].index('_')
    end = fname[i].index('.')
    trueAngle.append(int(fname[i][beg+1:end]))
    im.append(plt.imread(fname[i]))
    gray.append(cv2.cvtColor(im[i],cv2.COLOR_BGR2GRAY))
    dataHSV.append(cv2.cvtColor(im[i],cv2.COLOR_BGR2HSV))
    HOG[i] = hog(gray[i],orientations=9,pixels_per_cell=(10,10),
                    cells_per_block=(1,1),visualise=False)
    histH[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[0]
    histS[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[1]
    #histV[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[2]
    #feat[i] = np.hstack((HOG[i],histH[i],histS[i],histV[i]))

feat = np.column_stack((HOG,histH,histS))

scale.fit(feat)
scaleFeat = scale.transform(feat)

predict = ANN.predict(scaleFeat)
predictAngle = np.degrees(np.arctan2(predict[:,0],predict[:,1]))
predictAngle[np.where(predictAngle < 0)] += 360
error = abs(trueAngle-predictAngle)
error[error > 350] -= 360
error = abs(error)
MAE = np.mean(error)
print('MAE = ', MAE)
maxError = max(error)
print('Max Error = ', maxError)
trueAngle1 = np.asarray(trueAngle)
trueAngle1.sort()
plt.figure(3)
plt.clf()
plt.plot(trueAngle1,error)
plt.xlabel('Ground Truth Angle')
plt.ylabel('Error')

f,ax1 = plt.subplots(15,24)
itr = -1
#arrowLength = 100
#triangle = 5
#ax = f.gca()
for j in range(ax1.shape[0]):
    for k in range(ax1.shape[1]):
        itr += 1
        ax1[j,k].set_yticklabels([])
        ax1[j,k].set_xticklabels([])
        #plt.figure(4)
        #plt.subplot(15,24,im)
        ax1[j,k].imshow(im[itr])
        # = fig.gca()
        #ax.arrow(100,100,np.cos(predictAngle[0])*50,-np.sin(predictAngle[0])*50,fc='c',ec='m',head_width=5,head_length=15,linewidth=5)
        ax1[j,k].arrow(100,100,predict[itr][1]*60,-predict[itr][0]*60,fc='m',ec='m',head_width=20,head_length=15,linewidth=3)
        '''
        ax1[j,k].plot([100,100+predict[itr][1]*arrowLength],[100,100-predict[itr][0]*arrowLength],'m',linewidth=2)
        if predictAngle[itr] >= 315 and predictAngle[itr] < 360:
            ax1[j,k].plot(100+predict[itr][1]*arrowLength,100-predict[itr][0]*arrowLength,'>m',markersize=triangle)
        if predictAngle[itr] >= 0 and predictAngle[itr] < 45:
            ax1[j,k].plot(100+predict[itr][1]*arrowLength,100-predict[itr][0]*arrowLength,'>m',markersize=triangle)
        if predictAngle[itr] >= 45 and predictAngle[itr] < 135:
            ax1[j,k].plot(100+predict[itr][1]*arrowLength,100-predict[itr][0]*arrowLength,'^m',markersize=triangle)
        if predictAngle[itr] >= 135 and predictAngle[itr] < 225:
            ax1[j,k].plot(100+predict[itr][1]*arrowLength,100-predict[itr][0]*arrowLength,'<m',markersize=triangle)
        if predictAngle[itr] >= 225 and predictAngle[itr] < 315:
            ax1[j,k].plot(100+predict[itr][1]*arrowLength,100-predict[itr][0]*arrowLength,'vm',markersize=triangle)
        '''
'''
fig = plt.figure(5)

plt.clf()
ax = fig.gca()
#ax.arrow(10,5,20,-20,fc='c',ec='m',head_width=5,head_length=15,linewidth = 3)
plt.imshow(im[45]) 
ax.arrow(100,100,predict[45][1]*75,-predict[45][0]*75,fc='c',ec='m',head_width=5,head_length=15,linewidth=5)
plt.axis('equal')
#plt.xlim(0,80)
#plt.ylim(-40,10)
#plt.subplot(15,24,im)
'''   
#f, ax = plt.subplots(15,24)    
#ax[0,0].imshow(im[0])
#ax[0,0].plot([100,100+predict[j][1]*75],[100,100-predict[j][0]*75],'m')
        
