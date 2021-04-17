import streamlit as st
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.title("Diabetes classification - SVM")
img = plt.imread("causes-of-diabetes.webp")
st.image(img, caption="Diabetes")

x0 = st.sidebar.number_input("Pregnancies (months)", min_value=0)
x1 = st.sidebar.number_input("Glucose (mg/dL)", min_value=0)
x2 = st.sidebar.number_input("Blood Pressure", min_value=0)
x3 = st.sidebar.number_input("Skin Thickness", min_value=0)
x4 = st.sidebar.number_input("Insulin", min_value=0.0)
x5 = st.sidebar.number_input("BMI (kg/m2)", min_value=0.0)
x6 = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0)
x7 = st.sidebar.slider("Age", min_value=0)


model = joblib.load('svm_model.pkl')
if st.sidebar.button("Get result"):
    sample = np.array([x0, x1, x2, x3, x4, x5, x6, x7]).reshape(1, -1)
    pred = model.predict(sample)
    if pred == 0:
        st.write('**The person is not diabetic**')
    else:
        st.write('**The person is diabetic**')