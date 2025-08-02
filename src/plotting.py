# src/plotting.py

import plotly.express as px
import pandas as pd

def plot_traffic_volume_over_time(df):
    hourly_traffic = df.groupby('Hour')[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].mean().reset_index()
    fig = px.line(
        hourly_traffic, 
        x='Hour', 
        y=['CarCount', 'BikeCount', 'BusCount', 'TruckCount'],
        title='Average Vehicle Counts per Hour',
        labels={'value': 'Average Vehicle Count', 'Hour': 'Hour of Day'},
        template='plotly_white'
    )
    return fig

def plot_day_wise_traffic(df):
    day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['DayName'] = df['DayOfWeekNum'].map(day_map)
    fig = px.box(
        df, 
        x='DayName', 
        y='Total',
        color='DayName',
        title='Total Traffic Distribution by Day of the Week',
        labels={'Total': 'Total Vehicles', 'DayName': 'Day of the Week'},
        category_orders={"DayName": list(day_map.values())},
        template='plotly_white'
    )
    return fig

def plot_feature_importance(model, features):
    importance_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Traffic Prediction',
        template='plotly_white'
    )
    return fig