# Binary Classification 
 Binary classification of the data ```train.txt``` and test the performance on ```validation.txt```
# Install
 - Anaconda3 distribution for  Python3 https://www.anaconda.com/distribution/
and install requirment.txt
```
pip install -r requirment.txt
```
 - or use python 3 and install this requirment.txt using the above command also 
- jupyter notebook (for better understanding of the steps and how i made these decisions).

# File Structure

 * [model_code (code of developing  and testing the model)](./dir2) 
   * [preprocessing.py](./dir2/file21.ext)
   * [preprocessing.ipynp](./dir2/file22.ext)
   * [Training.ipynp](./dir2/file23.ext)
   * [Training.py](./dir2/file23.ext)
   * [training_processed.csv](./dir2/file23.ext)
   * [training.csv](./dir2/file23.ext)
   * [validation_processed.csv](./dir2/file23.ext)
   * [validation.csv](./dir2/file23.ext)

   
 * [README.md](./README.md)
 * [run.py (main code for run the micro service)](./README.md)
 * [prediction.py (prediction code)](./README.md)
 * [request.py (test cases to test microservice)](./README.md)
 * [label_encoder.pckl (label encoder model)](./README.md)
 * [one_hot.pckl (one hot encoder model)](./README.md)
 * [pca.pckl(pca model)](./README.md)
 * [scaler.pckl (scaling model)](./README.md)
 * [model.pckl (support vector machine model (SVC))](./README.md)






# Overview 

- ## Preprocessing Notebook Or preprocessing.py 
  - Both have the same code but the notebook have also data exploration code of the data sets and how i came up with these decision. 
  - ### Preprocessing steps

    - Read Data correctly (read variable2,3,8 14,15,17 and 19 correctly as numeric values)
    - imputation of variable1,4,5,6,7 with mode value (because they are categorial variables)
    - imputation of variable2,14,17 with mean value (because they are numeric variables)
    - Remove variable18 columns (because its have null values greater than 50% and also has strong correlation with variable10)
    - Remove variable5 because it is totally correlated with variable4
    - Remove variable17 because it is totally correlated with variable14
    - Remove variable19 because it can lead to missclassification (totally correlated with Class label in train data set but not in test data set)
    - Because the data train is unbalanced and this can led to high FalsePositive or high FalseNegative so we will use SMOTE technique to generate synthetic samples
    - Because the data have many categorial features and we must encode them to number so i will encode them using oneHotencoding technique (better than label encoding because the categorial data we have don't have order relationship)
     - save the data processed in ``` training_processed.csv``` and ```validation_processed.csv```
 

- ## Training Notebook Or training.py 
  - Both have the same code 
  - ### trainig steps

    - Loading the data
    - Splitting the data to x_train , y_train , x_test , y_test
    - Data Normalization
    - PCA and thats for two reason
      - PCA can be used to reduce dimensios but in case of our data it isn't problem (because after doing OneHotEncodeing the number of columns is 43 so it is not large number)
      - So i used PCA to reduce noise in the data by focusing on Princdipale components and remove components with low varaince (this in some cases can decrease accuracy because we lose some information but in our case accuracy have been improved ) 
    - Use Support Vector Machine as the classifier (SVM)
      - Use gaussian as the kernel because i think the data can be non linear separble 
      - choose class:weight 0 to be 2 to give more importance to minority class to try also reduce the effect of data inbalancing
      - using grid search technique try to search for best combinatio of c and gamma parameters of SVM and with some hand searching i choosed gamma = 0.0004 and c=0.1 (meaning of these values that  that i want the model to give the priority to choose good hyperplane rather than classify correctly on train data (To prevent overfitting))
    - ### Performance Criteria
         - To judge the classifier i will not take intp account accuracy because it can be misleading because the data is unbalanced i will compute f1 scores (because f1 score take into account recall and precision for specific class ) for both classes and take the average and this is my performance criteria 

- # Results  
```
Model Scores  
TotalNumber of tests 200  
TP 83  
TN 95  
FP 12  
FN 10  
Accuracy : 0.89  


*** Positive class - majority in training***  
Recall_True Posiive Rate : 0.8924731182795699  
Precision_True Positive Rate : 0.8736842105263158  
F1 positive : 0.8829787234042553  


*** Negative class - minority in training***   
Recall_True Negative Rate :(TN/(TN+FP))  0.8878504672897196  
Precision_True Negative Rate : # (TN/(TN+FN)) 0.9047619047619048  
F1 Negative : 0.8962264150943396   


************ Avg F1 score 0.8896025692492975    ***************
```
## Micro service 
To use the micro service you have two options 
1.  * python run.py  (this to run server locally)
    * Python request.py local 
1.  * Python request.py global  (to use the global server deployed on pythonanywhere host)
### Output 
```
**** for the first test case *****
<Response [200]>
{'output': 'no'}
**** for the second  test case *****
<Response [200]>
{'output': 'yes'}
```

## Notes 
### please ignore the warning   
  FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
If you want the future behaviour and silence this warning, you can specify "categories='auto'".
In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
  warnings.warn(msg, FutureWarning)

  



