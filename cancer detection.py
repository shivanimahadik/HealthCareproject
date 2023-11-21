# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:26:20 2023

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
 
cancer = pd.read_csv(r'C:\Users\Administrator\Downloads\HeathCare-project-main\Cancer_Data (1).csv')
cancer.head()

cancer.shape

cancer.isnull().sum()

cancer.drop(['Unnamed: 32','id'],axis = 1, inplace = True)

cancer.duplicated().sum()

cancer.info()

cancer.describe()

cancer['diagnosis'].value_counts()

label_encoder = LabelEncoder()
cancer['diagnosis'] = label_encoder.fit_transform(cancer['diagnosis'])

plt.figure(figsize = (10,6))
plt.pie(cancer['diagnosis'].value_counts(),autopct = '%1.2f%%',startangle = 90)
plt.title('Benign VS Malignant')
plt.show()

plt.figure(figsize=(20,20))



corr_matrix = cancer.corr()
correlation_with_target = abs(corr_matrix['diagnosis']).sort_values(ascending=False)
selected_features = correlation_with_target[1:16].index

X = cancer[selected_features]
y = cancer['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_classifier = LogisticRegression(random_state=42)
predictor = log_classifier.fit(X_train, y_train)
log_y_pred = predictor.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, log_y_pred)
accuracy_log_reg
confusion_mat = confusion_matrix(y_test, log_y_pred)
classification_rep = classification_report(y_test, log_y_pred)
print(confusion_mat)
print(classification_rep)

import pickle
pickle_out = open('cancer_classifier.pkl','wb')
pickle.dump(log_classifier,pickle_out)
pickle_out.close()

model = pickle.load(open('cancer_classifier.pkl','rb'))
#result = model.predict([0.2654,184.60,0.14710,25.38,122.80,2019.0,17.99,1001.0,0.3001,0.7119,0.27760,0.6656,1.095,8.589,153.4])

input_data = np.array([0.1288, 99.7, 0.04781, 15.11, 87.46, 711.2, 13.54, 566.3, 0.06664
, 0.1288, 0.08129, 0.1773, 0.2699, 2.058, 23.56])

# Reshape the input to a 2D array
input_data = input_data.reshape(1, -1)

result = model.predict(input_data)
result