# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:53:02 2019

@author: songl
"""

import submission as dt
import numpy as np
def a1():
    dataset = dt.load_csv('challenge_train.csv',class_index=0)
    train_features,train_classes= dataset
    sums=0
    for _ in range(10):
        tree2=dt.ChallengeClassifier()
        tree2.fit(train_features,train_classes)
        a=tree2.classify(train_features)
        for i in range(len(train_classes)):
            if a[i]==train_classes[i]:
                sums+=1
    print(sums/10/len(train_classes))

def a2():
    dataset = dt.load_csv('challenge_train.csv',class_index=0)
    train_features,train_classes= dataset
    sums=0
    for _ in range(5):
        tree2=dt.RandomForest(10,5,0.8,0.8)
        tree2.fit(train_features,train_classes)
        a=tree2.classify(train_features)
        for i in range(len(train_classes)):
            if a[i]==train_classes[i]:
                sums+=1
    print(sums/5/len(train_classes))

if __name__=='__main__':
    a1()
    