import streamlit as st
import pandas as pd
from pycaret.regression import *

# Set page configurations
st.set_page_config(
    page_title="Singapore  Resale Flat Prices Prediction",
    page_icon="🏠",
    layout="wide",  # Maximize the app size
    initial_sidebar_state="expanded"  # Always open the sidebar
)

# Streamlit UI
st.title(' Singapore  Resale Flat Prices Prediction') 

# Load the pre-trained model
loaded_model = load_model('Blend_pipeline')

df = pd.read_parquet('Data.parquet')
del df['Unnamed: 0']
del df['resale_price']
del df['price_per_square_meter']

# st.dataframe(df.head())

def get_user_input():
    user_input = {}
    for column in df.columns:
        if df[column].dtype == 'object' :  # Check if the column has categorical values
            user_input[column] = st.sidebar.selectbox(f'Select value for {column}:', df[column].unique())
        elif  df[column].dtype == 'category':
            user_input[column] = st.sidebar.selectbox(f'Select value for {column}:', df[column].unique())
        else:
           # Use a slider for numerical columns
            min_value = df[column].min()
            max_value = df[column].max()
            user_input[column] = st.sidebar.slider(f'Choose value for {column}:', min_value, max_value, min_value)
    return user_input
# Add input elements
user_input = get_user_input()


user_input_df = pd.DataFrame(user_input, index=[0])

# Make predictions
if st.button('Make Prediction'):
    # input_data = [user_input[column] if df[column].dtype == 'object' else float(user_input[column]) for column in df.columns]
    st.write('User Input:')
    st.dataframe(user_input_df)
    predictions = predict_model(loaded_model, data=user_input_df)
    st.write('Prediction:', predictions['prediction_label'])
