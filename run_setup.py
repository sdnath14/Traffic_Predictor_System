# run_setup.py

import os
from src.data_preprocessing import process_data
from src.model_pipeline import train_and_save_model

if __name__ == "__main__":
    # --- Configuration ---
    # Ensure your raw data file is named this and is inside the 'data' folder.
    input_filename = 'traffic_raw_data.csv'
    
    input_path = os.path.join('data', input_filename)
    output_path = 'data/processed_traffic_data.csv'
    model_path = 'models/random_forest_classifier.joblib'

    # --- Directory and File Checks ---
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists(input_path):
        print(f"Error: Raw data file not found at '{input_path}'")
        print("Please make sure your CSV file is in the 'data' directory.")
    else:
        # --- Run Pipelines ---
        print("Starting data preprocessing...")
        process_data(input_path, output_path)
        print(f"Data processed and saved to '{output_path}'")

        print("\nStarting model training...")
        train_and_save_model(output_path, model_path)
        print(f"Model trained and saved to '{model_path}'")

        print("\nSetup complete! You can now run the Streamlit app.")

