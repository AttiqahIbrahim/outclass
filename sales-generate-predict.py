import pickle #load model
import streamlit as st
import pandas as pd
import seaborn as sns

st.write("# Sales Prediction App")
st.write("This app predicts the **Sales based on Advertising type!**")

st.sidebar.header('User Input Parameters')

#(min,max,default present)
def user_input_features():
    TV = st.sidebar.slider('TV', 0.0, 296.4, 149.75)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 22.9)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 114.0, 25.75)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features
            
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingModel.h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(df) # testing (examination)


st.subheader('Prediction')
st.write(new_pred)

