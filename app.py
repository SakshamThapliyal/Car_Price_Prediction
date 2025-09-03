import streamlit as st
import pandas as pd
from joblib import load

# Set up the app page
st.set_page_config(page_title="Car Price Prediction", page_icon=":car:", layout="wide")
st.title('Car Price Prediction App')

st.markdown("""
This app predicts the price of a used car
based on the entered specifications using a trained machine learning model.
""")

st.sidebar.header('Input Car Features')  # Sidebar section for feature input

# Load trained car price model pipeline (must handle preprocessing internally)
model = load("./Car_Price_Prediction.joblib")

# Define feature choices (edit lists if needed for your dataset)
manufacturers = [
    'TOYOTA', 'HONDA', 'CHEVROLET', 'LEXUS', 'FORD', 'HYUNDAI', 'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP',
    'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC',
    'FIAT', 'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ', 'CITROEN', 'LAND ROVER', 'MINI',
    'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC', 'PEUGEOT', 'BENTLEY',
    'VOLVO', 'HAVAL', 'HUMMER', 'SCION', 'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH',
    'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE', 'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL'
]
categories = ['Hatchback', 'Sedan', 'Jeep', 'Coupe', 'Wagon', 'Convertible']
leather_options = ['Yes', 'No']
fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'CNG']
gear_types = ['Automatic', 'Tiptronic', 'Variator', 'Manual']
drive_types = ['Front', '4x4', 'Rear']
door_options = ['2', '4', '6']  # As strings (object dtype)
wheel_options = ['Left', 'Right']

# Feature input widgets
manufacturer = st.selectbox('Manufacturer', manufacturers, index=0)
prod_year = st.number_input('Production Year', min_value=1939, max_value=2025, value=2014, step=1)
category = st.selectbox('Category', categories, index=0)
leather_interior = st.selectbox('Leather Interior', leather_options, index=0)
fuel_type = st.selectbox('Fuel Type', fuel_types, index=0)
mileage = st.number_input('Mileage (km)', min_value=0, max_value=2147483647, value=91901, step=1000)
cylinders = st.number_input('Cylinders', min_value=2.0, max_value=16.0, value=4.0, step=1.0)
gear_box_type = st.selectbox('Gear Box Type', gear_types, index=0)
drive_wheels = st.selectbox('Drive Wheels', drive_types, index=0)
doors = st.selectbox('Doors', door_options, index=1)
wheel = st.selectbox('Wheel', wheel_options, index=0)
airbags = st.number_input('Airbags', min_value=0, max_value=16, value=4, step=1)
engine_size = st.number_input('Engine Size (L)', min_value=0.6, max_value=20.0, value=1.3, step=0.1)
is_turbo = st.selectbox('Turbocharged', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=0)

predict = st.button('Predict Car Price')

if predict:
    # Collect all user inputs into a DataFrame for model input
    input_data = pd.DataFrame([{
        'Manufacturer': manufacturer,
        'Prod. year': int(prod_year),
        'Category': category,
        'Leather interior': leather_interior,
        'Fuel type': fuel_type,
        'Mileage': int(mileage),
        'Cylinders': float(cylinders),
        'Gear box type': gear_box_type,
        'Drive wheels': drive_wheels,
        'Doors': doors,
        'Wheel': wheel,
        'Airbags': int(airbags),
        'engine_size': float(engine_size),
        'is_turbo': int(is_turbo)
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Car Price: ${int(prediction):,}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

