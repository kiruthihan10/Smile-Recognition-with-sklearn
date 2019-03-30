# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import neccesary libraries
import scipy
import sklearn as sk
import matplotlib.pyplot as plt
from PIL import Image
import numpy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


def img_loader(): #to read image files and store in 1D numpy array
    id_add = open("ALL_LIST_ppm.txt",'r')
    images = []
    for line in id_add:
        grand_row = []
        img = Image.open("faces/"+line.strip())
        img= (numpy.array(img))
        #print img[0][0]
        for row in img:
            for pixel in row:
                for color in pixel:
            #print row
                    grand_row.append(color)
        images.append(grand_row)
    return images

def smile_check():#By reading the files differentiate the pictures
    smile = open("SMILE_list_ppm.txt",'r')
    id_add = open("ALL_LIST_ppm.txt",'r')
    smile_list = []
    for line in smile:
        smile_list.append(line.strip())
    smiles = []
    x=0
    for name in id_add:
        smiled = False
        x+=1
        if name.strip() in smile_list:
            smiles.append(True)
        else:
            smiles.append(False)    
    return smiles
  
def with_num(y):#Change the logical values to int values
    out = []
    for i in y:
        if i == True:
            out.append(1)
        else:
            out.append(0)
    return out
          
def main():#Main code
    images = img_loader()
    smiles = smile_check()
    train_X = images[:900]
    test_X = images[900:]
    train_Y = smiles[:900]
    test_Y = smiles[900:]
    train_Y_with_num = with_num(train_Y)
    test_y_with_num = with_num(test_Y)
    
    #SVM method
    #Create Pipline
    svm_clf = Pipeline((("Scaler",StandardScaler()),("linear_svc",LinearSVC(C=1, loss="hinge")),))
    svm_clf.fit(train_X,train_Y)
    test = svm_clf.predict(test_X)
    n = 0
    correct = 0
    for i in test:
        if i == test_Y[n]:
            correct+=1
        n+=1
    print (float(correct)*100/len(test))
    
    svm_score = cross_val_score(svm_clf,train_X,train_Y,cv=3,scoring="accuracy")
    print (svm_score)
    
    y_train_pred = cross_val_predict(svm_clf,train_X,train_Y,cv=3)
    print (confusion_matrix(train_Y,y_train_pred))
    
    #SGD method
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(train_X,train_Y)
    test = sgd_clf.predict(test_X)

    
    
    
    n = 0
    correct = 0
    for i in test:
        if i == test_Y[n]:
            correct+=1
        n+=1
    print (float(correct)*100/len(test))
    sgd_score = cross_val_score(sgd_clf,train_X,train_Y,cv=3,scoring="accuracy")
    print (sgd_score)
    
    y_train_pred = cross_val_predict(sgd_clf,train_X,train_Y,cv=3)
    print (confusion_matrix(train_Y,y_train_pred))
    
    
print ("Running")
main()


