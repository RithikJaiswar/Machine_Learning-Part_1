import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title('Heart_Stroke_Prediction')
st.markdown('Provide Following Details')

age=st.slider('Age',18,100,40)
sex=st.selectbox('SEX',['Male','Female'])
chestpain = st.selectbox('Chest Pain Type',['ATA','NAP','TA','ASY'])
resting_BP = st.number_input('Resting Blood pressure (mm Hg)',80,200,120)
cholesterol = st.number_input('Cholesterol (mm/dl)',100,600,200)
FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl',[0,1])
Resting_ECG = st.selectbox('Resting ECG',['Normal','ST','LVH'])
max_hr = st.slider('Max Heart rate',60,220,150)
exercise_enigma = st.selectbox('Exercise-Induced Angina',['Y','N'])
oldpeak = st.slider('Oldpeak (ST Depression)',0.0,6.0,1.0)
st_slope = st.selectbox('ST Slope',['Up','Flat','Down'])

if st.button('Predict'):
    raw_input = {
        'Age' : age,
        'Resting_BP' : resting_BP,
        'Cholesterol' : cholesterol,
        'FastingBS' : FastingBS,
        'Resting_ECG' + Resting_ECG:1,
        'Max_HR' : max_hr,
        'Exercise_induced_angina' : exercise_enigma,
        'Oldpeak' : oldpeak,
        'ST_slope' + st_slope : 1,
        'Sex' + sex:1,
        'ChestPainType' + chestpain :1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.write('Heart Stroke Detected')
    else:
        st.write('Heart Stroke Not Detected')



