# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:28:08 2023

@author: Administrator
"""
# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Load the trained cancer model
cancer_model = pickle.load(open('cancer_classifier.pkl', 'rb'))

# Define the feature names for cancer prediction
CANCER_FEATURES = [
    'concave points_worst', 'perimeter_worst', 'concave points_mean',
    'radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean',
    'area_mean', 'concavity_mean', 'concavity_worst', 'compactness_mean',
    'compactness_worst', 'radius_se', 'perimeter_se', 'area_se'
]

# Load the diabetes prediction model
diabetes_model = pickle.load(open('diabetes_classifier.pkl', 'rb'))
# Define the feature names for diabetes prediction
DIABETES_FEATURES = ['age', 'bmi','hypertension','heart_disease','smoking_history','gender','HbA1c_level','blood_glucose_level']  # Add more features as needed


#Load the heart prediction model
heart_model = pickle.load(open('heart_classifier.pkl', 'rb'))
HEART_FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cancer')
def cancer():
    return render_template('cancer.html', FEATURES=CANCER_FEATURES)

@app.route('/cancer_predict', methods=['POST'])
def cancer_predict():
    if request.method == 'POST':
        input_data = [float(request.form[feature]) for feature in CANCER_FEATURES]

        # Reshape the input data to a 2D array
        input_data = np.array(input_data).reshape(1, -1)

        # Make predictions
        result = cancer_model.predict(input_data)
        probability = cancer_model.predict_proba(input_data)

        # Interpret the result
        if result[0] == 1:
            prediction = 'Malignant'
        else:
            prediction = 'Benign'

        return render_template('cancer.html', prediction=prediction, probability=probability[0][1], FEATURES=CANCER_FEATURES)

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html', FEATURES=DIABETES_FEATURES)

@app.route('/diabetes_predict', methods=['POST'])
def diabetes_predict():
    if request.method == 'POST':
        # Get input data from the form
        input_data = {
            'age': float(request.form['age']),
            'bmi': float(request.form['bmi']),
            'hypertension': float(request.form['hypertension']),
            'heart_disease': float(request.form['heart_disease']),
            'smoking_history': request.form['smoking_history'],
            'gender': request.form['gender'],
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level'])
        }

       
        # Reshape the input data to a 2D array
        input_data_array = np.array(list(input_data.values())).reshape(1, -1)

        # Make predictions using the diabetes model
        diabetes_prediction = diabetes_model.predict(input_data_array)
        result = 'Diabetes' if diabetes_prediction[0] == 1 else 'Normal'

        return render_template('diabetes.html', result=result, FEATURES=DIABETES_FEATURES)

@app.route('/heart')
def heart():
    return render_template('heart.html', FEATURES=HEART_FEATURES)

@app.route('/heart_predict', methods=['POST'])
def heart_predict():
    if request.method == 'POST':
        # Get input data from the form
        input_data = {
            'Age': float(request.form['Age']),
            'Sex': request.form['Sex'],
            'ChestPainType': request.form['ChestPainType'],
            'RestingBP': float(request.form['RestingBP']),
            'Cholesterol': float(request.form['Cholesterol']),
            'FastingBS': float(request.form['FastingBS']),
            'RestingECG': request.form['RestingECG'],
            'MaxHR': float(request.form['MaxHR']),
            'ExerciseAngina': request.form['ExerciseAngina'],
            'Oldpeak': float(request.form['Oldpeak']),
            'ST_Slope': request.form['ST_Slope']
        }
        
        # Reshape the input data to a 2D array
        input_data_array = np.array(list(input_data.values())).reshape(1, -1)

        # Make predictions using the heart disease model
        heart_disease_prediction = heart_model.predict(input_data_array)

        # Convert the prediction to a human-readable format
        result = 'Heart Disease' if heart_disease_prediction[0] == 1 else 'No Heart Disease'

        return render_template('heart.html', result=result, FEATURES=HEART_FEATURES)



if __name__ == '__main__':
    app.run(debug=True)

