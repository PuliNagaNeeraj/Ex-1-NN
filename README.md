<H3>ENTER YOUR NAME: Puli Naga Neeraj</H3>
<H3>ENTER YOUR REGISTER NO: 212223240130</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 10-3-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### Import Libraries
```
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
### Read the dataset
```
df = pd.read_csv('Churn_Modelling.csv')
print(df)
```
### values of X & Y
```
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:, -1].values
print(y)
```
### Check for outliers
```
df.describe()
```
### Missing Values
```
print(df.isnull().sum())
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
y = df.iloc[:, -1].values
print(y)
```
### Check for duplicates
```
df.duplicated()
```
### Droppimg string values from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
### Normalize dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
### Training and testing model
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
### 
## OUTPUT:
### Reading Data
![image](https://github.com/user-attachments/assets/a1fdd870-6289-4ae7-ab5e-635c03d38654)

### X features & Y features
![image](https://github.com/user-attachments/assets/9ebc6f12-79b7-4a45-90ac-2e65b0fc8db9)

### Outliers
![image](https://github.com/user-attachments/assets/671743b4-b87c-43e8-b4b2-f6ff6d586d0d)

### Missing Values
![image](https://github.com/user-attachments/assets/95c55d7c-c627-433c-9216-1daf8f64ceff)

### Duplicates Identification
![image](https://github.com/user-attachments/assets/44967d22-8ae5-4a11-9801-30a59341ffb1)

### Normalize the dataset
![image](https://github.com/user-attachments/assets/7bb8cb39-05b7-46c9-ac96-e88b2b8d30cc)

### SPLITTING THE TRAINING AND TESTING DATASET
![image](https://github.com/user-attachments/assets/b6f67a61-2822-41dd-8634-c67c3ab31401)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


