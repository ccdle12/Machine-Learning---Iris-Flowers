# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:41:45 2017

@author: USER
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

irisDataSet = load_iris();
                       
setosa = irisDataSet.target_names[0];
versicolor = irisDataSet.target_names[1];
virginica = irisDataSet.target_names[2];                                     

#Creating Decision Tree Classifier
#Feed decision tree the dataset and the related targets to each data set
clf = tree.DecisionTreeClassifier();
clf.fit(irisDataSet.data, irisDataSet.target);

for i in range(len(irisDataSet.target)):
    prediction = clf.predict(irisDataSet.data[i]);
                            
    if prediction == 0:
        print ("Prediction: %s | Position: %d " % (setosa, i));
    elif prediction == 1:
        print ("Prediction: %s | Position: %d " % (versicolor, i));
    else:
        print ("Prediction: %s | Position: %d " % (virginica, i));
              
print (clf.predict(irisDataSet.data[145]));