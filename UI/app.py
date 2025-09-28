import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("best_model.pkl")  # saved pipeline (preprocessing + model)

st.title("Heart Disease Prediction App")

st.sidebar.header("User Input Features")

def user_input():
    # Key inputs from user
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (0=Female, 1=Male)", (0, 1))
    trestbps = st.sidebar.number_input("Resting BP", 80, 200, 120)
    chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
    thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)

    # Other features → default values
    cp = 0          # chest pain type
    fbs = 0         # fasting blood sugar > 120 mg/dl
    restecg = 1     # resting ECG normal
    exang = 0       # no exercise induced angina
    slope = 1       # slope of ST segment
    ca = 0          # number of major vessels
    thal = 2        # thalassemia (normal)

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
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
