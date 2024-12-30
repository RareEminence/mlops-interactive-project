import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_iris

# Get the current working directory
current_dir = os.getcwd()

# Use relative path to the model within the app folder
model_path = os.path.join(current_dir, 'app', 'iris_model.pkl')

# Check if the model path exists to avoid errors
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    # Load the trained model
    model = joblib.load(model_path)

    # Set up the app interface
    st.title("🌸 Interactive MLOps App: Iris Flower Prediction")

    # Sidebar inputs
    st.sidebar.header("Input Features")
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    # Prepare user input for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(features)
    prediction_prob = model.predict_proba(features)

    # Display prediction results
    st.write("### Prediction")
    st.write(f"The predicted class is: **{prediction[0]}**")
    
    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame(prediction_prob, columns=load_iris().target_names)
    st.bar_chart(prob_df.T)

    # Monitoring metrics (Placeholders, replace with actual logic as needed)
    st.write("### Monitoring Metrics")
    st.metric(label="Response Time (ms)", value=50)  # Placeholder
    st.metric(label="Total Requests", value=10)  # Placeholder
