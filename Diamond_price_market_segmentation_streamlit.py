import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

try: #Loading the models
    lrmodel = joblib.load('Diamond_price_prediction_best_model.joblib')
    clmodel = joblib.load('kmeans_model_diamond_market_seg.joblib')
    scaler = joblib.load('kmeans_scaler_diamond_market_seg.joblib')
except Exception as e:
    st.error(f' Error in loading model: {e}' )
    st.stop()
               #Function for deriving the new features out of input data
def derived_features(carat,x,y,z,cut,color,clarity):
    cut_encoded = cut_map[cut]
    color_encoded = color_map[color]
    clarity_encoded = clarity_map[clarity]
    volume = x*y*z
    size_index = (x+y+z)/3
    quality_score = cut_encoded + color_encoded + clarity_encoded
    log_carat = np.log1p(carat)
    density = carat / volume
    dimension_ratio = x / y
    symmetry = min(x, y, z) / max(x, y, z)
    return volume,size_index,quality_score,log_carat,density,dimension_ratio,symmetry,clarity_encoded


#Mapping - Ordinal encoding
cut_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}

color_map = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7 }

clarity_map = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8 }


cluster_names = {
    0: 'Luxury Large Diamonds',
    1: 'Premium Small Diamonds',
    2: 'Low-quality Budget Diamonds',
    3: 'Mid-range Diamonds'
}
    

st.title("Diamond Price Prediction & clustering (Market Segmentation)") 

with st.form("Diamond Features Form"):#Form for getting inputs
    col1, col2 = st.columns(2)

    with col1:
        carat = st.number_input("Carat",min_value=0.10,max_value=1.0, value=0.5, step = 0.01)
        x = st.number_input("Length of diamond(mm)",min_value=3.0,max_value=10.0, value=5.0, step = 0.01)
        y = st.number_input("Width of diamond(mm)",min_value=3.0,max_value=10.0, value=5.0, step = 0.01)
        z = st.number_input("Height of diamond(mm)",min_value=1.0,max_value=6.0, value=5.0, step = 0.01)
        table = st.number_input("Table(width of top surface in %)",min_value=50.0,max_value=65.0, value=55.0, step = 0.1)
        cut = st.selectbox("Cut Quality", options=list(cut_map.keys()), index=0,help="Quality of the diamond's cut")
        color = st.selectbox("Color Grade", options = list(color_map.keys()), index=0,help="Color grade of diamond")
        clarity = st.selectbox("Clarity Grade", options = list(clarity_map.keys()), index=0,help="Clarity grade of diamond")
        volume,size_index,quality_score,log_carat,density,dimension_ratio,symmetry,clarity_encoded = derived_features(carat,x,y,z,cut,color,clarity)
        
           
    with col2:
        st.write(f'The volume is:{volume}')
        st.write(f'The symmetry is:{symmetry}')
        st.write(f'The quality score is: {quality_score}')

    action = st.radio("Choose action",['Predict price', 'Market Segmentation'])

    submit = st.form_submit_button("SUBMIT")
    #Functions for arranging input data
def regression_input_data():
    lrinput_data = np.array([[carat, table, clarity_encoded, volume, dimension_ratio,
    size_index, quality_score, density, symmetry, log_carat]])
    return lrinput_data
    
def cluster_input_data():
    clinput_data = pd.DataFrame([[log_price, price_per_carat, log_carat, quality_score]], 
                                columns=['log_price', 'price_per_carat', 'log_carat', 'quality_score'])
    return clinput_data

    #Functions for feeding data to models
def predict_price(lrinput_data):
    predicted_usd_price = lrmodel.predict(lrinput_data)[0]
    price_inr = predicted_usd_price * 83
    price_per_carat = predicted_usd_price / carat
    log_price = np.log1p(predicted_usd_price)
    return predicted_usd_price,price_inr,price_per_carat,log_price

def clustering(clinput_data):
    scaled = scaler.transform(clinput_data)
    cluster = clmodel.predict(scaled)[0]
    cluster_label = cluster_names[cluster]
    return cluster, cluster_label
        
#Decision making for choosing the model
if submit:
    try:
        if action == 'Predict price':
            lrinput_data = regression_input_data()
            #st.write(lrinput_data)
            predicted_usd_price,price_inr,price_per_carat,log_price = predict_price(lrinput_data)
            st.write(f'USD PRICE:    {predicted_usd_price:.2f}      \n     INR PRICE:    {price_inr:.2f}')

        elif action == 'Market Segmentation':
            lrinput_data = regression_input_data()
            predicted_usd_price,price_inr,price_per_carat,log_price = predict_price(lrinput_data)
            clinput_data = cluster_input_data()
            cluster, cluster_label = clustering(clinput_data)
            st.write(f'CLUSTER:    {cluster}       \n     CLUSTER LABEL:    {cluster_label}')    

    except Exception as e:
        st.error(f"Prediction failed: {e}")    