
import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from model_code.Preprocessing import objectsToNumbers,labelEncoding,oneHotEncoding,imputeWithMean,imputeWithMode

class predictor():
  def __init__(self):
    with open('model.pckl', 'rb') as handle:
        self.model = pickle.load(handle)
    with open('scaler.pckl', 'rb') as handle:
        self.scaler = pickle.load(handle)
    with open('pca.pckl', 'rb') as handle:
        self.pca = pickle.load(handle)
    
    self.train=pd.read_csv("model_code/training.csv",sep=';')
    columns=["variable2" ,"variable3","variable8"]
    objectsToNumbers(columns,self.train)
    columns=["variable1","variable4","variable5","variable6","variable7"]
    imputeWithMode(columns,self.train)
    columns=["variable14","variable17","variable2"]
    imputeWithMean(columns,self.train)
    columns = ["variable5","variable17","variable18","variable19"]
    self.train.drop(columns, axis=1, inplace=True)


  def preprocessing(self,features):
    columns=["variable2" ,"variable3","variable8"]
    objectsToNumbers(columns,features)
    columns = ["variable5","variable17","variable18","variable19"]
    features.drop(columns, axis=1, inplace=True)
    print ("Dealing with missing values and unwanted columns finished")

    train=self.train.copy()
    categoryColumns=labelEncoding(train,features,prediction=True)
    train,features,categoryColumns= oneHotEncoding(train,features,categoryColumns,"classLabel")
    train.drop(categoryColumns, axis=1, inplace=True)
    features.drop(categoryColumns, axis=1, inplace=True)
    return features



  def predict(self, features):
    print(features)
    columns = ["variable%d" % (i + 1) for i in range(15)]+["variable17","variable18","variable19"]
    features=pd.DataFrame(data=[features],columns = columns)
    features=self.preprocessing(features)
    features = self.scaler.transform(features)
    features = self.pca.transform(features)
    return self.model.predict(features)


binary_predictor=predictor()