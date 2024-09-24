import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Drop the 'Outcome' and 'Diabetes Pedigree Function' columns from features
X = data.drop(['Outcome', 'DiabetesPedigreeFunction'], axis=1)  # Adjust the column name as per your dataset
y = data['Outcome']

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Train the model
gbm_model = GradientBoostingClassifier(random_state=42)
gbm_model.fit(X_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter the following details to check if you are diabetic:")

# Input fields for user data
pregnancy = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=300, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Button for prediction
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = np.array([[pregnancy, glucose, blood_pressure, skin_thickness, insulin, bmi, age]])
    # input_data_scaled = scaler.transform(input_data)  # Scale the input data
    prediction = gbm_model.predict(input_data)

    # Display the result
    if prediction == 1:
        st.write("**Result:** Yes, you are diabetic.")
        st.image("soory.jpeg", caption="Sorry, you are diabetic.")
    else:
        st.write("**Result:** No, you are not diabetic.")
        st.image("its all good.jpeg", caption="All good, you are not diabetic.")

# # Display model accuracy (optional)
# accuracy = accuracy_score(y_test, gbm_model.predict(X_test))
# st.write(f"Model Accuracy: {accuracy:.2f}")
