import streamlit as st
import pandas as pd
from pycaret.regression import *

# Streamlit UI
st.title(' Singapore  Resale Flat Prices Prediction') 

# Load the pre-trained model
loaded_model = load_model('XGBOOST_pipeline')

df = pd.read_parquet('Data.parquet')
del df['Unnamed: 0']
df['town'] = df.town.astype('category')
df['flat_type'] = df.flat_type.astype('category')
df['storey_range'] = df.storey_range.astype('category')
df['age_of_flat'] = df.age_of_flat.astype('int8')
df['resale_price'] = df.resale_price.astype('float32')
df['lease_commence_date'] = df.lease_commence_date.astype('int16')
st.dataframe(df.head())

def get_user_input():
    user_input = {}
    for column in df.columns:
        if df[column].dtype == 'object' :  # Check if the column has categorical values
            user_input[column] = st.sidebar.selectbox(f'Select value for {column}:', df[column].unique())
        elif  df[column].dtype == 'category':
            user_input[column] = st.sidebar.selectbox(f'Select value for {column}:', df[column].unique())
        else:
            user_input[column] = st.sidebar.text_input(f'Enter value for {column}:', 0.0)
    return user_input
# Add input elements
user_input = get_user_input()


# Make predictions
if st.button('Make Prediction'):
    input_data = [user_input[column] if df[column].dtype == 'object' else float(user_input[column]) for column in df.columns]
    st.write(input_data)
    predictions = predict_model(loaded_model, data=input_data)
    st.write('Prediction:', predictions)
