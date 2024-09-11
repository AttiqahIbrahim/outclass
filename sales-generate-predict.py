import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

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

data = sns.load_dataset('Sales')
X = data.drop(['Sales'],axis=1)
Y = data.Sales.copy()
#X=df_scaled.drop('Sales', axis=1)
#y=df_scaled.Sales.copy()

modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Sales for each Advertising type')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
