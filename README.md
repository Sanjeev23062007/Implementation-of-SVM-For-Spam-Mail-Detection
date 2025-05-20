## Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Step-1
Import the packages.

## Step-2
Analyse the data.

## Step-3
Use modelselection and Countvectorizer to preditct the values.

## Step-4
Find the accuracy and display the result.

## Program:
```python


#Program to implement the SVM For Spam Mail Detection..
#Developed by: Sanjeev A
#RegisterNumber: 212224230246

import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)


```

## Output:

![Screenshot 2025-05-14 161106](https://github.com/user-attachments/assets/670925a6-ec95-42b2-ba98-cc657bca7108)

![Screenshot 2025-05-14 161114](https://github.com/user-attachments/assets/3f5ec01e-a8d2-47dc-8998-84e0b1a7d10e)

![Screenshot 2025-05-14 161117](https://github.com/user-attachments/assets/22f6d7a5-f815-4405-aa73-824bcbe6e0da)

![Screenshot 2025-05-14 161129](https://github.com/user-attachments/assets/e10af387-bbe6-4412-b5da-790dd7d5300e)

![Screenshot 2025-05-14 161135](https://github.com/user-attachments/assets/f3b07459-85c4-4bb9-bb36-382980acd506)

![Screenshot 2025-05-14 161141](https://github.com/user-attachments/assets/693ce7e3-c8d6-44ef-9d0b-f33be3ecbc18)

![Screenshot 2025-05-14 161146](https://github.com/user-attachments/assets/ae812844-f040-4f7b-93d1-186fd42b682b)

![Screenshot 2025-05-14 161150](https://github.com/user-attachments/assets/102d075a-1fd8-4b47-b95f-29411334da30)


![Screenshot 2025-05-14 161154](https://github.com/user-attachments/assets/40559975-a46d-4b04-bdd7-d3988b5253ee)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
