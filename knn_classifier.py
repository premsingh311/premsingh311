# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:47:15 2018

@author: deepikapantola
"""

# Assigning features and label variables
# First Feature
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
# Second Feature
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

# Label or target varible
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

target=['yes','no']

##Encoding data columns
# Import LabelEncoder

from sklearn import preprocessing
from sklearn import metrics
#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)

# converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
label
#combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))
features
#performing Knn classification
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=10)

# Train the model using the training sets
model.fit(features,label)


test=[[1,2],[2,2]]
#Predict Output
predicted= model.predict(test) # 0:Overcast, 2:Mild
predicted

from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
#_______________________________________________________________________________________
#IRIS DATA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv(r'E:\dataset\\User_Data.csv')
X=df.iloc[:,2:4]
y=df.iloc[:,4]


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

knn= KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
y_predicted=knn.predict(X_test)
y_predicted
accuracy_score(y_predicted,y_test)
