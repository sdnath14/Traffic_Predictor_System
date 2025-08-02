🚦 Traffic Flow Prediction Dashboard
An end-to-end machine learning project that predicts urban traffic situations using a Random Forest model and visualizes historical data in an interactive web application.

Live Application: https://traffic-predictor-system.onrender.com/

📋 Project Overview
This project tackles the challenge of urban traffic congestion by leveraging machine learning. It uses a Random Forest Classifier trained on a historical traffic dataset to predict traffic conditions (categorized as Low, Normal, High, or Heavy) with 94.5% accuracy.

The model is deployed in an interactive web application built with Streamlit, which allows users to get real-time predictions by inputting various parameters and to explore historical data through insightful visualizations.

✨ Features
High-Accuracy Model: Utilizes a Random Forest Classifier for reliable and accurate predictions.
Interactive Prediction Interface: A user-friendly dashboard for real-time traffic forecasting based on user inputs.
Exploratory Data Analysis (EDA): A dedicated section with Plotly charts to visualize historical traffic patterns by hour and day of the week.
Feature Importance: Displays which factors (like car count or hour) most influence the model's predictions.
End-to-End Workflow: The project covers the complete pipeline from data preprocessing and feature engineering to model training and deployment.
🛠️ Tech Stack
Language: Python 3.11
Libraries:
Web Framework: Streamlit
Data Manipulation: Pandas
Machine Learning: Scikit-learn
Numerical Operations: NumPy
Data Visualization: Plotly
Model Persistence: Joblib
Deployment: Render
⚙️ Setup and Running Locally
To run this project on your local machine, follow these steps:
Clone the repository:
git clone https://github.com/sdnath14/Traffic_Predictor_System.git
cd Traffic_Predictor_System

Install the required dependencies:
pip install -r requirements.txt

Run the setup script (one-time only):
This script processes the raw data and trains the machine learning model.
python run_setup.py

Launch the Streamlit application:
streamlit run app.py

The application will open in your web browser at http://localhost:8501.
📂 File Structure
The project is organized with a clear and modular structure:

Traffic_Predictor_System/
│
├── 📂 data/                 # Contains raw and processed datasets
│   ├── traffic_raw_data.csv
│   └── processed_traffic_data.csv
│
├── 📂 models/               # Stores the trained model file
│   └── random_forest_classifier.joblib
│
├── 📂 src/                  # Source code for modular functions
│   ├── data_preprocessing.py
│   ├── model_pipeline.py
│   └── plotting.py
│
├── 📜 app.py                # Main Streamlit application script
├── 📜 requirements.txt      # Project dependencies
├── 📜 runtime.txt           # Specifies Python version for deployment
└── 📜 run_setup.py         # Automation script for setup


