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

#Print a data set with its target label                            
print (irisDataSet.data[0]);
print (irisDataSet.target[0]);
      
# We need to remove 1 of each type of flower from the data set so that 
# we can test the classifier with data it has never seen                                    
removedVariablesFromIrisDataSet = [0, 50, 100];

# Training data                                      
# we are removing the removedVariablesFromIrisDataSet from the datset 
# The training data will contain ALL the data EXCEPT the removedVariablesFromIrisDataSet
training_target = np.delete(irisDataSet.target, removedVariablesFromIrisDataSet);                       
training_data = np.delete(irisDataSet.data, removedVariablesFromIrisDataSet, axis=0);
                      
# Testing Data
# This is the data we are going to test the classifier against as 'unseen' data
testing_target = irisDataSet.target[removedVariablesFromIrisDataSet];
testing_data = irisDataSet.data[removedVariablesFromIrisDataSet];                     
                               
#Creating Decision Tree Classifier
#Feed decision tree the dataset and the related targets to each data set
clf = tree.DecisionTreeClassifier();
clf.fit(training_data, training_target);

#if the testing_target is the same as the prediction - the machine has successfully predicted the correct flower       
print (testing_target);
print (clf.predict(testing_data));      

