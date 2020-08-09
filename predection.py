
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
    with open('label_encoder.pckl', 'rb') as handle:
      self.labels_encoder = pickle.load(handle)
    with open('one_hot.pckl', 'rb') as handle:
      self.one_hots = pickle.load(handle)


  def preprocessing(self,features):
    columns=["variable2" ,"variable3","variable8"]
    objectsToNumbers(columns,features)
    columns = ["variable5","variable17","variable18","variable19"]
    features.drop(columns, axis=1, inplace=True)


    categoryColumns=[]
    for (key, value) in self.one_hots:
      features[key] = self.labels_encoder[key].transform(features[key])
      x=value.transform(features[key].values.reshape(-1,1)).toarray()
      dfOneHot = pd.DataFrame(x, columns = [key+"_"+str(int(i)) for i in range(x.shape[1])])
      features = pd.concat([features, dfOneHot], axis=1)
      categoryColumns.append(key)

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