import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_flight_data(num_rows=5000):
    """
    Generate a dataset with flight parameters that influence landing quality.
    
    Parameters:
    num_rows (int): Number of data points to generate
    
    Returns:
    pd.DataFrame: Generated flight data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate base timestamps
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(hours=i*4) for i in range(num_rows)]
    
    # Aircraft types
    aircraft_types = ['Boeing 737', 'Airbus A320', 'Bombardier CRJ', 'Embraer E190', 'Boeing 787']
    
    # Weather conditions
    weather_conditions = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Windy', 'Foggy', 'Stormy']
    
    # Generate data with correlations to landing quality
    data = {
        'timestamp': timestamps,
        'aircraft_type': np.random.choice(aircraft_types, num_rows),
        'weather': np.random.choice(weather_conditions, num_rows),
        'runway_length_ft': np.random.uniform(5000, 12000, num_rows),
        'visibility_miles': np.random.uniform(0.5, 10, num_rows),
        'wind_speed_knots': np.random.uniform(0, 35, num_rows),
        'wind_direction_degrees': np.random.uniform(0, 359, num_rows),
        'crosswind_component_knots': np.random.uniform(0, 25, num_rows),
        'approach_speed_knots': np.random.uniform(120, 160, num_rows),
        'descent_rate_fpm': np.random.uniform(500, 1200, num_rows),
        'altitude_ft': np.random.uniform(0, 1000, num_rows),
        'glideslope_deviation_dots': np.random.uniform(-2, 2, num_rows),
        'localizer_deviation_dots': np.random.uniform(-2, 2, num_rows),
        'pitch_degrees': np.random.uniform(-5, 15, num_rows),
        'roll_degrees': np.random.uniform(-10, 10, num_rows),
        'flap_setting_degrees': np.random.choice([15, 20, 25, 30, 35, 40], num_rows),
        'gear_position': np.random.choice(['Down', 'In Transit', 'Up'], num_rows, p=[0.95, 0.03, 0.02]),
        'distance_from_threshold_feet': np.random.uniform(0, 5000, num_rows),
        'throttle_percentage': np.random.uniform(20, 70, num_rows),
    }
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable: landing_rate (vertical speed at touchdown)
    # Lower (more negative) values indicate harder landings
    
    # Base landing rate with some randomness
    base_landing_rate = -np.random.normal(300, 150, num_rows)
    
    # Factors that affect landing rate
    # Poor conditions lead to harder landings (more negative landing rate)
    
    # High descent rate → harder landing
    base_landing_rate -= (df['descent_rate_fpm'] - 700) * 0.3
    
    # High approach speed → harder landing
    base_landing_rate -= (df['approach_speed_knots'] - 140) * 0.8
    
    # Glideslope deviation → harder landing when off glideslope
    base_landing_rate -= abs(df['glideslope_deviation_dots']) * 50
    
    # Localizer deviation → harder landing when off centerline
    base_landing_rate -= abs(df['localizer_deviation_dots']) * 40
    
    # Low visibility → harder landing
    base_landing_rate -= (10 - df['visibility_miles']) * 5
    
    # Crosswind → harder landing with strong crosswind
    base_landing_rate -= df['crosswind_component_knots'] * 3
    
    # Weather effects
    weather_effect = pd.Series(0, index=range(num_rows))
    weather_effect[df['weather'] == 'Heavy Rain'] -= 50
    weather_effect[df['weather'] == 'Stormy'] -= 100
    weather_effect[df['weather'] == 'Windy'] -= 70
    weather_effect[df['weather'] == 'Foggy'] -= 60
    base_landing_rate += weather_effect
    
    # Add some noise
    base_landing_rate += np.random.normal(0, 50, num_rows)
    
    # Clip to realistic values
    landing_rate = np.clip(base_landing_rate, -800, -50)
    df['landing_rate_fpm'] = landing_rate
    
    # Create binary target: hard landing (typically defined as < -300 fpm)
    df['hard_landing'] = (df['landing_rate_fpm'] < -400).astype(int)
    
    # Add a risk score (continuous scale from 0-100%)
    # This represents the "ground truth" risk which our ML model will try to predict
    df['risk_score'] = 1 / (1 + np.exp(-0.02 * (-df['landing_rate_fpm'] - 300)))
    df['risk_score'] = df['risk_score'] * 100  # Convert to percentage
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/flight_data.csv', index=False)
    
    print(f"Generated {num_rows} rows of flight data")
    print(f"Number of hard landings: {df['hard_landing'].sum()} ({df['hard_landing'].mean()*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    generate_flight_data(5000)
