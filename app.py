import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.plotting import plot_traffic_volume_over_time, plot_day_wise_traffic, plot_feature_importance
import os

st.set_page_config(
    page_title="Traffic Flow Prediction",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model_and_data():
    model_path = 'models/random_forest_classifier.joblib'
    data_path = 'data/processed_traffic_data.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        st.error("Model or data file not found. Please run `python run_setup.py` first.")
        return None, None
        
    artifacts = joblib.load(model_path)
    data = pd.read_csv(data_path)
    return artifacts, data

artifacts, data = load_model_and_data()

if artifacts is not None:
    model = artifacts['model']
    features = artifacts['features']
    label_map = artifacts['label_map']

    st.title("🚦 Traffic Flow Prediction Dashboard")
    st.markdown("Predict traffic situations based on inputs and explore historical data patterns.")

    tab1, tab2 = st.tabs(["📈 Prediction", "📊 Exploratory Data Analysis"])

    with tab1:
        st.header("Predict Traffic Situation")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Features")
            hour = st.slider("Hour of the Day", 0, 23, 8)
            day_map_input = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            day_name = st.selectbox("Day of the Week", list(day_map_input.keys()))
            day_of_week_num = day_map_input[day_name]
            
            car_count = st.slider("Car Count", 0, 500, 150)
            bike_count = st.slider("Bike Count", 0, 500, 100)
            bus_count = st.slider("Bus Count", 0, 100, 20)
            truck_count = st.slider("Truck Count", 0, 100, 25)

        input_data = np.array([[hour, day_of_week_num, car_count, bike_count, bus_count, truck_count]])
        
        with col2:
            st.subheader("Prediction Result")
            if st.button("Predict Traffic", type="primary", use_container_width=True):
                prediction_encoded = model.predict(input_data)[0]
                prediction_label = label_map.get(prediction_encoded, "Unknown").capitalize()
                
                if prediction_label == 'Heavy':
                    st.error(f"Predicted Traffic Situation: **{prediction_label}**")
                elif prediction_label == 'High':
                    st.warning(f"Predicted Traffic Situation: **{prediction_label}**")
                elif prediction_label == 'Normal':
                    st.info(f"Predicted Traffic Situation: **{prediction_label}**")
                else: 
                    st.success(f"Predicted Traffic Situation: **{prediction_label}**")
            
            st.markdown("---")
            st.subheader("Model Feature Importance")
            fig_importance = plot_feature_importance(model, features)
            st.plotly_chart(fig_importance, use_container_width=True)

    with tab2:
        st.header("Analysis of Historical Traffic Data")
        
        st.markdown("### Average Vehicle Counts per Hour")
        st.plotly_chart(plot_traffic_volume_over_time(data), use_container_width=True)

        st.markdown("### Total Traffic Distribution by Day of the Week")
        st.plotly_chart(plot_day_wise_traffic(data), use_container_width=True)

else:
    st.info("Please set up the project by running the `run_setup.py` script.")