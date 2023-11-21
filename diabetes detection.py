# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:12:08 2023

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score

diabetes = pd.read_csv(r'C:\Users\Administrator\Downloads\HeathCare-project-main\diabetes_prediction_dataset.csv')
diabetes.head()

diabetes.shape
diabetes['smoking_history'].unique()
diabetes.drop_duplicates(inplace = True)
diabetes.duplicated().sum()

diabetes.info()
diabetes.describe()

#sns.countplot('diabetes',data = diabetes)

label_encoder = LabelEncoder()
diabetes['gender'] = label_encoder.fit_transform(diabetes['gender'])
diabetes['smoking_history'] = label_encoder.fit_transform(diabetes['smoking_history'])

X = diabetes.drop('diabetes', axis=1)
y = diabetes['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a classifier (Random Forest in this example)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred,labels=[1,0])
print(cm)

# Train a classifier (Random Forest in this example)
lg = LogisticRegression()
lg.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred = lg.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred,labels=[1,0])
print(cm)

import pickle
pickle_out = open('diabetes_classifier.pkl','wb')
pickle.dump(clf,pickle_out)
pickle_out.close()

model = pickle.load(open('diabetes_classifier.pkl','rb'))
ans = model.predict([[0,44.0,0,0,4,19.31,6.5,200]])
print(ans)



