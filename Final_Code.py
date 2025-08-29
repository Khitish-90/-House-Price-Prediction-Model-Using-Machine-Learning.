import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

# Load Dataset Function
@st.cache_data
def load_data():
    file_paths = [
        "kc_house_data.csv",
        r"C:\ML project 2\House_data.csv"
    ]
    for path in file_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error("Dataset file not found! Please check the file path.")
    return None

# Load Data
data = load_data()
if data is None:
    st.stop()

# Selecting relevant features
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'condition', 'grade']
X = data[features]
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Streamlit Navigation
st.sidebar.title("🏡 House Price Prediction Model")
page = st.sidebar.radio("Navigation", ["Home", "About", "Prediction"])

if page == "Home":
    st.title("Welcome")
    st.subheader("🏡 House Price Prediction Model")
    st.write("Use this model to predict house prices based on various features.")

elif page == "About":
    st.title("About")
    st.write("This application predicts house prices using a Linear Regression model trained on real estate data.")
    st.write("Developed with Streamlit and Scikit-Learn.")

elif page == "Prediction":
    st.title("Predict House Price")
    st.sidebar.header("Enter House Features")
    
    sqft_living = st.sidebar.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
    bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=5, value=2)
    floors = st.sidebar.number_input("Floors", min_value=1, max_value=3, value=1)
    condition = st.sidebar.slider("Condition (1-5)", 1, 5, 3)
    grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
    
    if st.sidebar.button("Predict Price"):
        input_data = np.array([[sqft_living, bedrooms, bathrooms, floors, condition, grade]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        st.success(f"🏠 Predicted House Price: **${prediction[0]:,.2f}**")
