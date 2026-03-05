import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load the encoders and scaler
with open("label_encoder.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Input")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active member", [0, 1])

# ... (your imports and loading code stay the same)

# 1. Prepare the initial numerical/label encoded data
# NOTE: We do NOT include Geography here yet because we encode it separately
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products], # Added this - it was missing from your dict!
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# 2. One-hot encode "Geography" using the raw 'geography' variable from the selectbox
# We pass [[geography]] to ensure it's a 2D array for the encoder
geo_encoded = onehot_encoder_geo.transform([[geography]])

# 3. Create the DataFrame for the encoded columns
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# 4. Combine them
# IMPORTANT: Ensure this order matches your training data columns!
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# 5. Scale and Predict
input_data_scaled = scaler.transform(input_df)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.subheader(f"Churn Probability: {prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")