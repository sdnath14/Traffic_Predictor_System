import pandas as pd
from sklearn.preprocessing import LabelEncoder

def process_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %I:%M:%S %p')

    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeekNum'] = df['DateTime'].dt.dayofweek

    label_encoder = LabelEncoder()
    df['TrafficSituationEncoded'] = label_encoder.fit_transform(df['Traffic Situation'])  
    features_to_keep = [
        'Hour', 'DayOfWeekNum', 'CarCount', 'BikeCount', 
        'BusCount', 'TruckCount', 'Total', 'TrafficSituationEncoded', 'Traffic Situation'
    ]
    df_processed = df[features_to_keep].copy()
    
    df_processed.to_csv(output_path, index=False)
    return df_processed
