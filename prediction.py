import streamlit as st
import joblib
import numpy as np

# Load saved models
scaler = joblib.load("C:/Users/theha/OneDrive/Attachments/Desktop/house price/scaler (3).pkl")
model = joblib.load("C:/Users/theha/OneDrive/Attachments/Desktop/house price/random_forest_model (2).pkl")

st.title("House Price Prediction App")
st.markdown("Enter the property features below to estimate the house price.")

# Layout with columns for better UI
col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("Square Footage", min_value=0, max_value=10000, value=2000)
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=1990)

with col2:
    lot_size = st.number_input("Lot Size", min_value=0, max_value=100000, value=5000)
    garage_size = st.number_input("Garage Size", min_value=0, max_value=10, value=2)
    neighborhood_quality = st.selectbox("Neighborhood Quality (1-10)", options=list(range(1, 11)))

# Prepare input for model
input_data = np.array([[square_footage, num_bedrooms, num_bathrooms, year_built,
                        lot_size, garage_size, neighborhood_quality]])

# Predict
if st.button("Predict Price"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Estimated House Price: ${prediction[0]:,2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
