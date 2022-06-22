# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Sasi Dharan
RegisterNumber:  212221240049


import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## Data Head:
![image](https://user-images.githubusercontent.com/94154712/175121954-62195d16-a5c2-410e-a010-2dab7d1cdcb6.png)
## Data Info:
![image](https://user-images.githubusercontent.com/94154712/175122020-7f634b74-40ac-4032-b779-7fb398574d73.png)
## Data isnull():
![image](https://user-images.githubusercontent.com/94154712/175122099-c59b722a-d53c-4ef2-ba5e-ff5a6c695d96.png)
![image](https://user-images.githubusercontent.com/94154712/175122168-0e0b076a-3f7c-48be-8d3b-a7eb5e7d332f.png)
## Accuracy:
![image](https://user-images.githubusercontent.com/94154712/175122244-0dd10d21-b76c-4f9e-89da-df7bd925e255.png)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
