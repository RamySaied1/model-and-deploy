
import pandas as  pd
import matplotlib as mpl
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score





def objectsToNumbers(columns,df):
    def convert(numStr):
        if type(numStr)is float :
            return numStr
        numStr=numStr.replace(',','.')
        return float(numStr)

    for col in columns:
        df[col]=df[col].apply(convert)       


def imputeWithMode(columns,df):
    for col in columns:
        df[col].fillna(df[col].mode()[0],inplace=True)
    
def imputeWithMean(columns,df):
    for col in columns:
        df[col].fillna(df[col].mean(),inplace=True)
        
    
def labelEncoding(train,test,prediction=False):
    categorical_feature_mask = train.dtypes==object
    categorical_cols = train.columns[categorical_feature_mask].tolist()    
    le=None
    print(categorical_cols)
    for col in categorical_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        print(col)
        if (prediction and col=="classLabel"):
            continue
        test[col] = le.transform(test[col])
    
    return categorical_cols 
    


def oneHotEncoding(train,test,categorical_cols,classLabel):
    sol=[]
    for col in categorical_cols:
        sol.append(len(train[col].unique()))
    
    categorical_cols.remove(classLabel)
    for col in categorical_cols:
        oe=OneHotEncoder(categories='auto')
        x=oe.fit_transform(train[col].values.reshape(-1,1)).toarray()
        dfOneHot = pd.DataFrame(x, columns = [col+"_"+str(int(i)) for i in range(x.shape[1])])
        train = pd.concat([train, dfOneHot], axis=1)
    
        x=oe.transform(test[col].values.reshape(-1,1)).toarray()
        dfOneHot = pd.DataFrame(x, columns = [col+"_"+str(int(i)) for i in range(x.shape[1])])
        test = pd.concat([test, dfOneHot], axis=1)
        
    
    return train,test,categorical_cols

    
def overSampling(train,classLabel):
    ## Smote
    oldColumns=train.columns
    oldColumns=oldColumns.drop(classLabel)
    oldColumns=oldColumns.append(pd.Index([classLabel]))
    sm = SMOTE(sampling_strategy='minority', random_state=7)
    oversampled_trainX, oversampled_trainY = sm.fit_sample(train.drop(classLabel, axis=1), train[classLabel])
    train = pd.concat([ pd.DataFrame(oversampled_trainX),pd.DataFrame(oversampled_trainY)], axis=1)
    train.columns = oldColumns
    
    return train


# # ********** ******** ************** **************
# # Preprocessing
# # ********** ******** ************** **************
# - Read Data correctly (read variable2,3,8 14,15,17 and 19 correctly as numeric values)
# - imputation of variable1,4,5,6,7 with mode value (because they are categorial variables)
# - imputation of variable2,14,17 with mean value (because they are numeric variables)
# - remove variable18 columns (because its have null values greater than 50% and also has strong correlation with variable10)
# - remove variable5 because it is totally correlated with variable4
# - remove variable17 because it is totally correlated with variable14
# - remove variable19 because it can lead to missclassification (totally correlated with Class label in train data set but not in test data set)
# - because the data train data is unbalanced and this can led to high FalsePositive or high FalseNegative so we will use SMOTE technique to generate synthetic samples (so size of training data set will increase)
# - because the data have many categorial features and we must encode them to number so i will encode them using oneHotencoding technique (better than label encoding because the categorial data we have don't have order relationship)
# 

def main():
    train = pd.read_csv("training.csv",sep=';')
    test = pd.read_csv("validation.csv",sep=';')

    ## read numerical data correctly
    columns=["variable2" ,"variable3","variable8"]
    objectsToNumbers(columns,train)
    objectsToNumbers(columns,test)

    ## missing value imputation
    columns=["variable1","variable4","variable5","variable6","variable7"]
    imputeWithMode(columns,train)
    imputeWithMode(columns,test)

    columns=["variable14","variable17","variable2"]
    imputeWithMean(columns,train)
    imputeWithMean(columns,test)    

    ## remove unwanted columns
    columns = ["variable5","variable17","variable18","variable19"]
    train.drop(columns, axis=1, inplace=True)
    test.drop(columns, axis=1, inplace=True)

    print ("Dealing with missing values and unwanted columns finished")


    ## label encoding
    categoryColumns=labelEncoding(train,test)


    ## one Hot Encodinf
    train,test,categoryColumns= oneHotEncoding(train,test,categoryColumns,"classLabel")

    ## remove original columns
    train.drop(categoryColumns, axis=1, inplace=True)
    test.drop(categoryColumns, axis=1, inplace=True)

    print("Encoding finished")


    # oversampling the training data
    train=overSampling(train,"classLabel")

    print("OverSampling (SMOTE) finished")


    # save the output
    train.to_csv("training_processed.csv",sep=';')
    test.to_csv("validation_processed.csv",sep=';')

    print ("\nPreprocessing finished \n")

