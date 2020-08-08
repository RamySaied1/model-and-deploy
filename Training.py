

import pandas as  pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import sys


def showMetrics(y_test,y_predictied):
    conMat=confusion_matrix(y_test,y_predictied)
    TP=conMat[1,1]
    TN=conMat[0,0]
    FP=conMat[0,1]
    FN=conMat[1,0]
    print ("-"*40)
    print ("Model Scores")
    print ("TotalNumber of tests",conMat.sum())
    print ("TP",TP)
    print ("TN",TN)
    print ("FP",FP)
    print ("FN",FN)

    print ("Accuracy :",accuracy_score(y_test,y_predictied))

    print ("\n\n*** Positive class - majority in training *** ")
    print ("Recall_True Posiive Rate :",recall_score(y_test,y_predictied)) # (TP/(TP+FN))
    print ("Precision_True Positive Rate :",precision_score(y_test,y_predictied)) #(TP/(Tp+FP))
    f1Positive=f1_score(y_test,y_predictied)
    print(  "F1 positive :",f1_score(y_test,y_predictied))
    
    
    print ("\n\n*** Negative class - minority in training *** ")
    RecallNegative=(TN/(TN+FP))
    print ("Recall_True Negative Rate :(TN/(TN+FP)) ",RecallNegative) # (TN/(TN+FP))
    precisionNegative=(TN/(TN+FN))
    print ("Precision_True Negative Rate : # (TN/(TN+FN))",precisionNegative)  # (TN/(TN+FN))
    f1Negative=2*((RecallNegative*precisionNegative)/(RecallNegative+precisionNegative))
    print(  "F1 Negative :", f1Negative  )
    
    print (    "\n\n\n ******************** Avg F1 score  ",  (f1Positive+f1Negative)/2,'    **************************')

    print ("-"*40)


    

# # ***** ***** ****** *****
# # Steps
# # ***** ***** ****** *****
# 
# - Loading the data
# - Splitting the data to x_train , y_train , x_test , y_test
# - Data Normalization
# - PCA and thats for two reason
#    - PCA can be used to reduce dimensios but in case of our data it isn't problem (because after doing OneHotEncodeing the number of columns is 43 so it is not large number)
#    - So i used PCA to reduce noise in the data by focusing on Princdipale components and remove components with low varaince (this in some cases can decrease accuracy because we lose some information but in our case accuracy have been improved ) 
# - Use Support Vector Machine as the classifier (SVM)
#   - USe gaussian as the kernel because i think the data can be non linear separble 
#   - choose class:weight 0 to be 2 to give more importance to minority class to try also reduce the effect of data imbalancing
#   - using grid search technique try to search for best combinatio of c and gamma parameters of SVM and with some hand searching i choosed gamma = 0.0004 and c=0.1 (meaning of these values that  that i want the model to give the priority to choose good hyperplane rather than classify correctly on train data (beacuase i belive have large noise))
# - To judge the classifier i will not take intp account accuracy because it can be misleading because the data is unbalanced i will compute f1 scores (because f1 score take into account recall and precision for specific class ) for both classes and take the average and this is my performance criteria 
#   
#    


train = pd.read_csv("training_processed.csv",sep=';')
test = pd.read_csv("validation_processed.csv",sep=';')

train.drop("Unnamed: 0", axis=1, inplace=True)
test.drop("Unnamed: 0", axis=1, inplace=True)



x_train=train.drop("classLabel",axis=1)
y_train=train["classLabel"]

x_test=test.drop("classLabel",axis=1)
y_test=test["classLabel"]

x_train.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print ("Normalization Finished\n")


from sklearn.decomposition import PCA
pca = PCA(27)                              # choosing by experiement
pca.fit(x_train)

x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

print("PCA Finished\n")


"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC # "Support Vector Classifier" 

parameter_candidates = [
  {'C': [1, 10, 100, 1000,1000], 'gamma': [0.001, 0.0004,0.00001], 'kernel': ['rbf']},
]

# Create a classifier object with the classifier and parameter candidates
clf = GridSearchCV(estimator=SVC(class_weight={0:2}), param_grid=parameter_candidates, n_jobs=-1,verbose=10)

# Train the classifier on data1's feature and target data
clf.fit(x_train, y_train)   
print('Best score for data1:', clf.best_score_) 
clf.best_estimator_

"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='rbf', class_weight={0:2},gamma=0.0004,C=0.1,tol=1e-9) 
#clf = SVC(kernel='rbf', class_weight={0:2},C=2000) 
      
# fitting x samples and y classes 
clf.fit(x_train, y_train)

print("Model Training Finished\n")


y_predictied=clf.predict(x_test)


showMetrics(y_test,y_predictied)

