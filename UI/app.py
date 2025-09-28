import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("best_model.pkl")  # saved pipeline (preprocessing + model)

st.title("Heart Disease Prediction App")

st.sidebar.header("User Input Features")

# Example input fields (adjust to your dataset)
def user_input():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", (0, 1))  # 0=female, 1=male
    chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)

    data = {
        "age": age,
        "sex": sex,
        "chol": chol,
        "trestbps": trestbps,
        "thalach": thalach,
        "oldpeak": oldpeak,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Prediction
prediction = model.predict(input_df)[0]
st.subheader("Prediction")
st.write("Heart Disease: ✅ Yes" if prediction == 1 else "Heart Disease: ❌ No")

# Bonus: simple visualization
st.subheader("Heart Disease Trends")
fig, ax = plt.subplots()
ax.bar(["No Disease", "Disease"], [60, 40])  # replace with dataset counts
st.pyplot(fig)
