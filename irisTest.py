# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:41:45 2017

@author: USER
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus;
from sklearn.externals.six import StringIO;
import condaGraphFix;

irisDataSet = load_iris();
                       
setosa = irisDataSet.target_names[0];
versicolor = irisDataSet.target_names[1];
virginica = irisDataSet.target_names[2]; 

#Print a data set with its target label                            
print ("Sample data set: {}".format(irisDataSet.data[0]));
print ("Target value of sample data set: {} | Name of sample data set: {}".format(irisDataSet.target[0], setosa));
      
# We need to remove 1 of each type of flower from the data set so that 
# we can test the classifier with data it has never seen before                                    
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
print ("Testing Target: {}".format(testing_target));
print ("Machine Predictions: {}".format(clf.predict(testing_data)));      
      
#Visualizing the decision tree        
dot_data = StringIO();
irisFlowerGraph = tree.export_graphviz(clf, 
                                       out_file = dot_data, 
                                       feature_names = irisDataSet.feature_names,
                                       class_names = irisDataSet.target_names,
                                       filled=True, rounded=True,
                                       impurity=False);


graph = pydotplus.graph_from_dot_data(dot_data.getvalue());
graph = condaGraphFix.fixGraph(graph);
graph.write_pdf("iris2.pdf");
                
#open("iris.pdf");

