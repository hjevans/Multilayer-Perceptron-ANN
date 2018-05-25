# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:21:42 2017

@author: Hannah
"""

import glob
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLP
import sklearn
import pickle

fname = glob.glob('roombas/train/*.jpg')
split = train_test_split(fname,test_size = .2,train_size = .8)
angleTrain = []
angleValid = []
imTrain = []
imValid = []
gray = []
dataHSV = []
HOG = [0]*len(split[0])
histH = [0]*len(split[0])
histS = [0]*len(split[0])
#histV = [0]*len(fname)
feat = [0]*len(split[0])
for i in range (len(split[0])):
    beg = split[0][i].index('_')
    end = split[0][i].index('.')
    angleTrain.append(int(split[0][i][beg+1:end]))
    imTrain.append(plt.imread(split[0][i]))
    gray.append(cv2.cvtColor(imTrain[i],cv2.COLOR_BGR2GRAY))
    dataHSV.append(cv2.cvtColor(imTrain[i],cv2.COLOR_BGR2HSV))
    HOG[i] = hog(gray[i],orientations=9,pixels_per_cell=(10,10),
                    cells_per_block=(1,1),visualise=False)
    histH[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[0]
    histS[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[1]
    #histV[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[2]
    #feat[i] = np.hstack((HOG[i],histH[i],histS[i],histV[i]))
gray1 = []
dataHSV1 = []
HOG1 = [0]*len(split[1])
histH1 = [0]*len(split[1])
histS1 = [0]*len(split[1])
#histV1 = [0]*len(split[1])
for j in range (len(split[1])):
    beg = split[1][j].index('_')
    end = split[1][j].index('.')
    angleValid.append(int(split[1][j][beg+1:end]))
    imValid.append(plt.imread(split[1][j]))
    gray1.append(cv2.cvtColor(imValid[j],cv2.COLOR_BGR2GRAY))
    dataHSV1.append(cv2.cvtColor(imValid[j],cv2.COLOR_BGR2HSV))
    HOG1[j] = hog(gray1[j],orientations=9,pixels_per_cell=(10,10),
                    cells_per_block=(1,1),visualise=False)
    histH1[j] = np.histogram(dataHSV1[j][:,:,0],bins=20,range=(0,255))[0]
    histS1[j] = np.histogram(dataHSV1[j][:,:,0],bins=20,range=(0,255))[1]
    #histV1[i] = np.histogram(dataHSV[i][:,:,0],bins=20,range=(0,255))[2]
    #feat[i] = np.hstack((HOG[i],histH[i],histS[i],histV[i]))
featTrain = np.column_stack((HOG,histH,histS))
featValid = np.column_stack((HOG1,histH1,histS1))

scaleTrain = StandardScaler()
scaleTrain.fit(featTrain)
scaleTrainFeat = scaleTrain.transform(featTrain)

scaleValid = StandardScaler()
scaleValid.fit(featValid)
scaleValidFeat = scaleValid.transform(featValid)

#scaleValidFeat = scaleTrain.transform(featValid)
radTrain = np.asarray(angleTrain)*np.pi/180
radValid = np.asarray(angleValid)*np.pi/180
sinTrain = np.sin(radTrain)
cosTrain = np.cos(radTrain)
sinValid = np.sin(radValid)
cosValid = np.cos(radValid)

trainLabel = np.column_stack((sinTrain,cosTrain)) # [0] is sin, [1] is cos
validLabel = np.column_stack((sinValid,cosValid))
#epoch = []
bestMAEs=[]
priorEpoch = 0
ANNs = []
bestANNs = []
#MAETrain = []
#MAEValid = []
#MAETrainAngle = []
#MAEValidAngle = []
numNodes = []
node = []
avgMAEs = []
bestMAE = float('inf')
history = np.array([float('inf')]*10)
itr = -1
priorValidAngles = []
for k in range(4,12,2):
    #ANN = MLP(hidden_layer_sizes = (k,),max_iter=30,activation='logistic',solver='lbfgs',warm_start=True)
    MAECurrent = []
    MAETrainCurrent = []
    for l in range(10):
        ANN = MLP(hidden_layer_sizes = (k,),max_iter=30,activation='logistic',solver='lbfgs',warm_start=True)
        #ANNs.append(ANN.fit(scaleTrainFeat,trainLabel))
        condition = True
        #set almost all lists to be empty here
        MAETrain = []
        MAEValid = []
        MAETrainAngle = []
        MAEValidAngle = []
        epoch = []
        loop = -1

        #itr = -1
        while(condition):
            itr += 1
            loop += 1
            #ANN = MLP(hidden_layer_sizes = (10,),max_iter=30,activation='logistic',solver='lbfgs', warm_start=False)
            ANNs.append(ANN.fit(scaleTrainFeat,trainLabel))
            #epoch.append(ANN.n_iter_)
            epoch.append(ANN.n_iter_ + priorEpoch)
            priorEpoch = epoch[loop]
            trainPredict = ANN.predict(scaleTrainFeat)
            validPredict = ANN.predict(scaleValidFeat)
            MAETrain.append(np.mean(abs(trainLabel[:,0]-trainPredict[:,0])) + np.mean(abs(trainLabel[:,1]-trainPredict[:,1])))
            MAEValid.append(np.mean(abs(validLabel[:,0]-validPredict[:,0])) + np.mean(abs(validLabel[:,1]-validPredict[:,1])))
            #MAETrain.append(np.mean(abs(trainLabel[0]-trainPredict[0])+abs(trainLabel[1]-trainPredict[1])))
            #MAEValid.append(np.mean(abs(validLabel[0]-validPredict[0])+abs(validLabel[1]-validPredict[1])))
            predTrainAngle = np.degrees(np.arctan2(trainPredict[:,0],trainPredict[:,1]))
            predValidAngle = np.degrees(np.arctan2(validPredict[:,0],validPredict[:,1]))
            predTrainAngle[np.where(predTrainAngle < 0)] += 360
            predValidAngle[np.where(predValidAngle < 0)] += 360
            errorValid = abs(angleValid-predValidAngle)
            errorValid[errorValid > 350] -= 360
            errorValid = abs(errorValid)
            errorTrain = abs(angleTrain-predTrainAngle)
            errorTrain[errorTrain > 350] -= 360
            errorTrain = abs(errorTrain)
            print(np.mean(errorTrain), np.mean(errorValid))
            MAETrainAngle.append(np.mean(errorTrain))
            MAEValidAngle.append(np.mean(errorValid))
            priorValidAngles.append(np.mean(errorValid))
            MAECurrent.append(np.mean(errorValid))
            MAETrainCurrent.append(np.mean(errorTrain))
            numNodes.append(k)
            
            plt.figure(1)
            plt.clf()
            plt.semilogy(epoch,MAETrainAngle,label='Training')
            plt.semilogy(epoch,MAEValidAngle,label='Validation')
            plt.xlabel('Epochs')
            plt.ylabel('Mean Absolute Error (degrees)')
            plt.legend()
            plt.pause(0.001)
            plt.savefig('figure1_%d.jpg' % loop)
            if MAEValidAngle[loop] < bestMAE:
                bestMAE = MAEValidAngle[loop]
            history[itr%10] = MAEValidAngle[loop]
            if np.all(history != bestMAE):
                condition = False
            elif np.all(history == bestMAE):
                condition = False
                #elif itr == 50:
                    #condition = False
        '''
        plt.figure(1)
        plt.clf()
        plt.semilogy(epoch,MAETrainAngle,label='Training')
        plt.semilogy(epoch,MAEValidAngle,label='Validation')
        plt.legend()
        plt.pause(0.001)
        '''
        bestMAEs.append(bestMAE)
        bestANNs.append(ANNs[max(np.where(priorValidAngles == bestMAE))[0]])
    '''
    plt.figure(1)
    plt.clf()
    plt.semilogy(epoch,MAETrainAngle,label='Training')
    plt.semilogy(epoch,MAEValidAngle,label='Validation')
    plt.legend()
    plt.pause(0.001)
    '''
    node.append(k)
    avgMAEs.append(np.mean(MAECurrent))
    
'''
plt.figure(1)
plt.clf()
plt.semilogy(epoch,MAETrainAngle,label='Training')
plt.semilogy(epoch,MAEValidAngle,label='Validation')
plt.legend()
#plt.pause(0.001)
'''
plt.figure(2)
plt.clf()
plt.plot(numNodes,priorValidAngles,'or',label='Per run',)
plt.plot(node,avgMAEs,'*c',markersize = 10,label='average')
plt.xlabel('Nodes')
plt.ylabel('MeanAbsolute Errors')
plt.legend()
plt.savefig('figure2.jpg')


bestestMAE = min(bestMAEs)
bestMAEs = np.asarray(bestMAEs)
bestestANN = bestANNs[max(np.where(bestMAEs==min(bestMAEs))[0])]

pickle.dump(bestestANN,open('ANN.pkl','wb'))
pickle.dump(scaleTrain,open('scale.pkl','wb'))        