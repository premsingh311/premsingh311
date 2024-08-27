# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:10:45 2024

@author: CDAC
"""

weather=['Sunny','Sunny','Overcast','Rainy','Rainy',
         'Rainy','Overcast','Sunny','Sunny','Rainy',
         'Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot', 'Mild','Cool','Cool','Cool','Mild','Cool','Mild', 'Mild', 'Mild'] 
      'Hot', 'Mild']
play=['No','No','Yes','Yes','Yes','No', 'Yes','No','Yes','Yes','Yes','Yes','Yes','No']


#2nd step
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

weather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

#combine weather and temperature to create a feature

feature=list(zip(weather_encoded,temp_encoded))

#import Gaussian Naive Bayes Model
from skelearn.Naive_ bayes import GaussianNB
#Create a Gaussian classifier
model = GaussianNB()
#Train the model using the traing sets
model.fit (features1, label)
#pedicted Output
predicted=Model.predict([[0,2]])#0:overcast,2:Mild
print"Predicted Value:",predicted
pridicted value:[1]