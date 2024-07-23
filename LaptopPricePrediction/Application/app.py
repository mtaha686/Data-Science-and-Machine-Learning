import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('laptoppricePredictor.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv('Preprocessed_laptop_data.csv')
print(data.columns)
# data.drop('Unnamed: 0', axis=1, inplace=True)

data['IPS'].unique()
st.title("laptop Price Predictor")

Company = st.selectbox("Brand", data['Company'].unique())
Type = st.selectbox("Type", data['TypeName'].unique())
Ram = st.selectbox('Ram(GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
Os = st.selectbox("OS", data['OpSys'].unique())
Weight = st.number_input("Weight of the laptop")
TouchScreen = st.selectbox("TouchScreen", ['Yes', 'No'])
Ips = st.selectbox("IPS", ['Yes', "No"])
screen_size = st.selectbox('Screen Size (inches)', [
                           11.6, 12.5, 13.3, 14, 15.6, 17.3])
Resolution = st.selectbox('Screen Resolution', [
    "1366x768",
    "1920x1080",
    "2560x1440",
    "3840x2160",
    "2880x1800",
    "1600x900",
    "1280x800"
]
)
Cpu = st.selectbox('CPU', data['CPU'].unique())
hdd = st.selectbox('HDD(GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(GB)', [0, 128, 256, 512, 1024])
gpu = st.selectbox('GPU (GB)', data['Gpu'].unique())

if st.button('Predict Price'):
    ppi = None
    if TouchScreen == "Yes":
        TouchScreen = 1
    else:
        TouchScreen = 0
    if Ips == "Yes":
        Ips = 1
    else:
        Ips = 0

    X_resolution = int(Resolution.split('x')[0])
    Y_resolution = int(Resolution.split('x')[1])

    ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

    query = np.array([Company, Type, Ram, Weight, TouchScreen,
                     Ips, ppi, Cpu, hdd, ssd, Os, gpu])
    query = query.reshape(1, 12)

    prediction = int(np.exp(rf.predict(query)[0]))

    st.title("Predicted price for this laptop could be between" +
             str(prediction-1000)+"rs" + "to" + str(prediction+1000)+"rs")
