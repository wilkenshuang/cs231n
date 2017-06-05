# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:28:15 2017

@author: wilkenshuang
"""

import numpy as np
import os
from collections import Counter

flist=['data_batch_1','data_batch_2','data_batch_3',
       'data_batch_4','data_batch_5','test_batch']

def get_path_file(filename):
    return os.path.join('cifar-10-batches-py/',filename)

def unpickle(file):
    import pickle
    with open(file,'rb') as f:
        dict=pickle.load(f,encoding='bytes')
        image=dict[b'data']
        cls=np.array(dict[b'labels'])
        image=image.reshape(10000,3,32,32).transpose([0,2,3,1])
    return image,cls

def load_data():
    images_train=[]
    cls_train=[]
    for i in range(5):
        images_batch,cls_batch=unpickle(get_path_file(flist[i]))
        images_train.append(images_batch)
        cls_train.append(cls_batch)
    Xtr=np.concatenate(images_train)
    Ytr=np.concatenate(cls_train)
    Xte,Yte=unpickle(get_path_file(flist[5]))
    return Xtr,Ytr,Xte,Yte

Xtr_row=np.zeros([50000,3072],dtype='int')
Xte_row=np.zeros([10000,3072],dtype='int')
Ytr=np.zeros([50000],dtype='int')
Yte=np.zeros([10000],dtype='int')

Xtr,Ytr,Xte,Yte=load_data()
#for i in range(2):
    #image,cls=unpickle(get_path_file(flist[i]))
    #Xtr_row[i*10000:(i+1)*10000]=image
    #Ytr[i*10000:(i+1)*10000]=cls

Xtr_row=Xtr.reshape([50000,32*32*3])
Xte_row=Xte.reshape([10000,32*32*3])

Xval_row=Xtr_row[49000:,:]
Yval=Ytr[49000:]
Xtr_row=Xtr_row[:49000,:]
Ytr=Ytr[:49000]

Xte_row,Yte=unpickle(get_path_file(flist[2]))

#Xtr_row=Xtr.reshape([50000,32*32*3])

te_row=Xte_row[1:5000]
te_labs=Yte[1:5000]
class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,X,y):
        self.xtr=X
        self.ytr=y
    def predict(self,X):
        num_test=X.shape[0]
        Ypred=np.zeros(num_test,dtype = self.ytr.dtype)
        for i in range(num_test):
            distances=np.sum(np.abs(self.xtr-X[i,:]),axis=1)
            #dictances=np.sqrt(np.sum(np.square(self.xtr-X[i,:]),axis=1))
            min_index=np.argmin(distances)
            Ypred[i]=self.ytr[min_index]
        return Ypred

class KNearestNeighbor(object):
    def __init__(self):
        pass
    def train(self,X,y):
        self.xtr=X
        self.ytr=y
    def predict(self,X,k):
        num_test=X.shape[0]
        Ypred=np.zeros(num_test,dtype = self.ytr.dtype)
        for i in range(num_test):
            #distances=np.sum(np.abs(self.xtr-X[i,:]),axis=1)
            distances=np.sqrt(np.sum(np.square(self.xtr-X[i,:]),axis=1))
            #min_index=np.argmin(distances)
            Distances=list(distances)
            pred=[]
            for j in range(k):
                min_index=Distances.index(min(Distances))
                pred.append(self.ytr[min_index])
                Distances.pop(min_index)
            Num=Counter(pred)
            Ypred[i]=np.array(Num.most_common(1))[0,0]
        return Ypred

validation=[]
for k in [1,3,5,10,20]:
    nn=KNearestNeighbor()
    nn.train(Xtr_row,Ytr)
    Yval_pred=nn.predict(Xval_row,k)
    #Yval_pred=nn.predict(Xval_row,k=k)
    #Yte_pred=nn.predict(te_row)
    print('accuracy: %f' %(np.mean(Yval_pred==Yval)))
    