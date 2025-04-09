import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pickle
import os
import time
from datetime import datetime

# Try to import the enhanced 3D aircraft visualization module
try:
    from assets.aircraft_3d import (
        add_aircraft_to_scene,
        create_wind_visualization,
        create_approach_guidance,
        create_animated_landing
    )
    ENHANCED_VISUALIZATION_AVAILABLE = True
except ImportError:
    ENHANCED_VISUALIZATION_AVAILABLE = False
    print("Could not import aircraft_3d module, falling back to basic visualization")

def load_models():
    """Load trained ML models"""
    try:
        with open('models/hard_landing_model.pkl', 'rb') as f:
            models = pickle.load(f)
        return models
    except FileNotFoundError:
        st.warning("Models not found. Training new models...")
        from model_trainer import train_models
        clf_model, reg_model = train_models()
        return {
            'classification': clf_model,
            'regression': reg_model
        }

def predict_landing_risk(input_data):
    """
    Predict hard landing risk based on input parameters
    
    Parameters:
    input_data (dict): Flight parameters for prediction
    
    Returns:
    tuple: (hard_landing_prediction, risk_score)
    """
    models = load_models()
    
    # Convert input data to DataFrame for prediction
    input_df = pd.DataFrame([input_data])
    
    # Make predictions
    try:
        hard_landing_prob = models['classification'].predict_proba(input_df)[0][1]
        risk_score = models['regression'].predict(input_df)[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return 0.0, 0.0
    
    return hard_landing_prob, risk_score

def create_aircraft_visualization(pitch, roll, altitude, risk_level, approach_speed=140, descent_rate=700, wind_speed=0, wind_direction=0):
    """
    Create visualization of aircraft orientation with enhanced 3D model
    
    Parameters:
    pitch (float): Pitch angle in degrees
    roll (float): Roll angle in degrees
    altitude (float): Altitude in feet
    risk_level (float): Risk level from 0 to 100
    approach_speed (float): Approach speed in knots
    descent_rate (float): Descent rate in feet per minute
    wind_speed (float): Wind speed in knots
    wind_direction (float): Wind direction in degrees
    
    Returns:
    plotly.graph_objects.Figure: Interactive visualization
    """
    # Create a figure with a transparent background
    fig = go.Figure()
    
    # Ground plane (runway)
    runway_length = 3000  # meters
    runway_width = 60  # meters
    
    # Scale altitude for visualization but ensure it's never negative
    altitude_scaled = max(min(altitude / 100, 40), 0.1)  # Scale altitude for visualization
    
    # Add runway with centerline and threshold markings
    # Main runway surface
    fig.add_trace(go.Mesh3d(
        x=[0, runway_length, runway_length, 0],
        y=[-runway_width/2, -runway_width/2, runway_width/2, runway_width/2],
        z=[0, 0, 0, 0],
        color='gray',
        opacity=0.7,
        showlegend=False
    ))
    
    # Define runway edge lights
    x_lights = []
    y_lights = []
    z_lights = []
    
    # Add runway edge lights every 60 meters
    for i in range(0, runway_length, 60):
        x_lights.extend([i, i])
        y_lights.extend([-runway_width/2, runway_width/2])
        z_lights.extend([0.5, 0.5])  # Slightly raised lights
    
    # Add touchdown zone markings
    for dist in range(300, 900, 150):
        # Left side markings
        fig.add_trace(go.Scatter3d(
            x=[dist, dist+30],
            y=[-runway_width/4, -runway_width/4],
            z=[0.1, 0.1],
            mode='lines',
            line=dict(color='white', width=10),
            showlegend=False
        ))
        
        # Right side markings
        fig.add_trace(go.Scatter3d(
            x=[dist, dist+30],
            y=[runway_width/4, runway_width/4],
            z=[0.1, 0.1],
            mode='lines',
            line=dict(color='white', width=10),
            showlegend=False
        ))
    
    # Add runway centerline (dashed)
    for i in range(0, runway_length, 30):
        if i % 60 == 0:  # Skip every other segment for dashed line
            fig.add_trace(go.Scatter3d(
                x=[i, i+20],
                y=[0, 0],
                z=[0.1, 0.1],
                mode='lines',
                line=dict(color='white', width=3),
                showlegend=False
            ))
    
    # Add threshold markings
    for i in range(-runway_width//2 + 5, runway_width//2, 10):
        fig.add_trace(go.Scatter3d(
            x=[100, 150],
            y=[i, i],
            z=[0.1, 0.1],
            mode='lines',
            line=dict(color='white', width=3),
            showlegend=False
        ))
    
    # Add approach lights
    approach_length = 900
    for i in range(-approach_length, 0, 150):
        fig.add_trace(go.Scatter3d(
            x=[i],
            y=[0],
            z=[0.5],
            mode='markers',
            marker=dict(color='white', size=8),
            showlegend=False
        ))
    
    # Environment elements - simplified horizon
    horizon_distance = 5000
    horizon_size = 10000
    
    # Add horizon/sky
    fig.add_trace(go.Mesh3d(
        x=[horizon_distance, horizon_distance, -horizon_distance, -horizon_distance],
        y=[-horizon_size, horizon_size, horizon_size, -horizon_size],
        z=[3000, 3000, 3000, 3000],
        color='skyblue',
        opacity=0.3,
        showlegend=False
    ))
    
    # Add distant terrain features
    # Hills
    hill_x = np.linspace(-horizon_size, horizon_size, 50)
    hill_y = np.linspace(-horizon_size, horizon_size, 50)
    hill_xx, hill_yy = np.meshgrid(hill_x, hill_y)
    hill_zz = np.zeros_like(hill_xx)
    
    # Create some terrain features with random heights
    np.random.seed(42)  # For reproducibility
    for i in range(5):
        center_x = np.random.uniform(-horizon_size/2, horizon_size/2)
        center_y = np.random.uniform(-horizon_size/2, horizon_size/2)
        height = np.random.uniform(200, 800)
        width = np.random.uniform(1000, 3000)
        hill_zz += height * np.exp(-((hill_xx-center_x)**2 + (hill_yy-center_y)**2) / (2*width**2))
    
    # Calculate aircraft position with runway approach perspective
    aircraft_x = runway_length / 3  # Position on approach
    aircraft_y = 0  # Centered - this might change based on roll
    aircraft_z = altitude_scaled

    # Calculate crosswind component
    crosswind = abs(wind_speed * np.sin(np.radians(wind_direction)))
    
    # Try to use the enhanced aircraft visualization if the module is loaded
    touchdown_point = None
    has_enhanced_viz = False
    try:
        if 'ENHANCED_VISUALIZATION_AVAILABLE' in globals() and ENHANCED_VISUALIZATION_AVAILABLE:
            has_enhanced_viz = True
            # Add detailed 3D aircraft model with position and orientation
            aircraft_position = (aircraft_x, aircraft_y, aircraft_z)
            aircraft_orientation = (pitch, roll, 0)  # Assuming yaw=0 for simplicity
            fig = add_aircraft_to_scene(fig, aircraft_position, aircraft_orientation, risk_level)
            
            # Add wind visualization if wind is present
            if wind_speed > 0:
                wind_position = (aircraft_x - 100, aircraft_y, aircraft_z + 20)
                fig = create_wind_visualization(fig, wind_speed, wind_direction, crosswind, wind_position)
            
            # Add approach guidance
            glideslope_dev = (altitude / (aircraft_x * np.tan(np.radians(3)))) - 1
            localizer_dev = aircraft_y / 10
            distance_from_threshold = aircraft_x
            fig = create_approach_guidance(fig, glideslope_dev, localizer_dev, distance_from_threshold)
            
            # Add landing path animation
            fig, touchdown_point = create_animated_landing(
                fig, pitch, roll, altitude, risk_level, approach_speed, descent_rate
            )
    except Exception as e:
        # Fallback to original simple visualization
        print(f"Using enhanced visualization failed: {e}")
        has_enhanced_viz = False
        
    # If enhanced visualization is not available or failed, use basic visualization
    if not has_enhanced_viz or touchdown_point is None:
        # Aircraft representation
        # Create a simplified aircraft shape
        aircraft_length = 40
        aircraft_wingspan = 40
        
        # Apply rotation matrices for pitch and roll
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Create nose and tail points
        nose_x = aircraft_x + aircraft_length/2 * np.cos(pitch_rad)
        nose_y = aircraft_y
        nose_z = aircraft_z + aircraft_length/2 * np.sin(pitch_rad)
        
        tail_x = aircraft_x - aircraft_length/2 * np.cos(pitch_rad)
        tail_y = aircraft_y
        tail_z = aircraft_z - aircraft_length/2 * np.sin(pitch_rad)
        
        # Create wingtip points with roll
        right_wing_x = aircraft_x
        right_wing_y = aircraft_y + aircraft_wingspan/2 * np.cos(roll_rad)
        right_wing_z = aircraft_z + aircraft_wingspan/2 * np.sin(roll_rad)
        
        left_wing_x = aircraft_x
        left_wing_y = aircraft_y - aircraft_wingspan/2 * np.cos(roll_rad)
        left_wing_z = aircraft_z - aircraft_wingspan/2 * np.sin(roll_rad)
        
        # Add vertical stabilizer
        vstab_top_x = tail_x
        vstab_top_y = aircraft_y
        vstab_top_z = tail_z + aircraft_length/4
        
        # Determine aircraft color based on risk level
        if risk_level < 30:
            aircraft_color = 'green'
        elif risk_level < 70:
            aircraft_color = 'orange'
        else:
            aircraft_color = 'red'
        
        # Add aircraft body (fuselage)
        fig.add_trace(go.Scatter3d(
            x=[nose_x, tail_x],
            y=[nose_y, tail_y],
            z=[nose_z, tail_z],
            mode='lines',
            line=dict(color=aircraft_color, width=8),
            showlegend=False
        ))
        
        # Add aircraft wings
        fig.add_trace(go.Scatter3d(
            x=[left_wing_x, right_wing_x],
            y=[left_wing_y, right_wing_y],
            z=[left_wing_z, right_wing_z],
            mode='lines',
            line=dict(color=aircraft_color, width=6),
            showlegend=False
        ))
        
        # Add vertical stabilizer
        fig.add_trace(go.Scatter3d(
            x=[tail_x, vstab_top_x],
            y=[tail_y, vstab_top_y],
            z=[tail_z, vstab_top_z],
            mode='lines',
            line=dict(color=aircraft_color, width=4),
            showlegend=False
        ))
        
        # Add reference lines for glideslope and approach path
        glideslope_x = np.linspace(0, runway_length * 3, 100)
        glideslope_z = glideslope_x * 0.05  # Approximately 3 degree glideslope
        
        fig.add_trace(go.Scatter3d(
            x=glideslope_x,
            y=np.zeros(len(glideslope_x)),
            z=glideslope_z,
            mode='lines',
            line=dict(color='lightgreen', width=2, dash='dash'),
            showlegend=False
        ))
        
        # Add simple landing path preview
        if altitude < 500:  # Only show landing path at lower altitudes
            # Simplified landing path based on current parameters
            path_x = np.linspace(aircraft_x, 0, 20)
            path_y = np.zeros(20) + aircraft_y * np.exp(-np.linspace(0, 5, 20))  # Converge to centerline
            path_z = np.maximum(0, aircraft_z * (1 - np.linspace(0, 1, 20)**2))  # Quadratic descent
            
            fig.add_trace(go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode='lines',
                line=dict(color=aircraft_color, width=2, dash='dot'),
                showlegend=False
            ))
    
    # Set up the layout
    fig.update_layout(
        title=None,
        scene=dict(
            xaxis=dict(title="Distance", showbackground=False, showticklabels=False, showgrid=False),
            yaxis=dict(title="Lateral", showbackground=False, showticklabels=False, showgrid=False),
            zaxis=dict(title="Altitude", showbackground=False, showticklabels=False, showgrid=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            aspectratio=dict(x=1.5, y=1, z=0.5)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
    )
    
    # Set scene background to dark
    fig.update_scenes(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        zaxis_showgrid=False,
        bgcolor='rgb(10,10,30)'
    )
    
    # Add landing risk level annotation
    risk_text = "Low Risk" if risk_level < 30 else "Medium Risk" if risk_level < 70 else "HIGH RISK"
    risk_color = "green" if risk_level < 30 else "orange" if risk_level < 70 else "red"
    
    fig.add_annotation(
        x=0.5,
        y=0.95,
        text=f"Landing Risk: {risk_text} ({risk_level:.1f}%)",
        showarrow=False,
        font=dict(size=14, color=risk_color),
        xref="paper",
        yref="paper"
    )
    
    # Add altitude annotation
    fig.add_annotation(
        x=0.05,
        y=0.05,
        text=f"Altitude: {altitude} ft",
        showarrow=False,
        font=dict(size=12, color="white"),
        xref="paper",
        yref="paper",
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )
    
    # Add approach speed annotation
    fig.add_annotation(
        x=0.05,
        y=0.10,
        text=f"Speed: {approach_speed} kts | Descent: {descent_rate} fpm",
        showarrow=False,
        font=dict(size=12, color="white"),
        xref="paper",
        yref="paper",
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )
    
    # Add wind annotation if applicable
    if wind_speed > 0:
        fig.add_annotation(
            x=0.05,
            y=0.15,
            text=f"Wind: {wind_speed} kts at {wind_direction}° | Xwind: {crosswind:.1f} kts",
            showarrow=False,
            font=dict(size=12, color="white"),
            xref="paper",
            yref="paper",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4
        )
    
    return fig

def create_risk_gauge(risk_score):
    """Create a gauge visualization for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hard Landing Risk"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
            'bar': {'color': "darkgrey"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig

def log_simulation_data(input_data, prediction_result):
    """Log simulation data for analysis"""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **input_data,
        'hard_landing_probability': prediction_result[0],
        'risk_score': prediction_result[1]
    }
    
    # Create log directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Append to log file
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv('logs/simulation_logs.csv', mode='a', header=not os.path.exists('logs/simulation_logs.csv'), index=False)
    
    return log_entry

def calculate_landing_parameters(approach_speed, descent_rate, pitch, risk_score):
    """
    Calculate parameters for landing results
    
    Parameters:
    approach_speed (float): Approach speed in knots
    descent_rate (float): Descent rate in feet per minute
    pitch (float): Pitch angle in degrees
    risk_score (float): Risk score from 0 to 100
    
    Returns:
    dict: Landing parameters including touchdown rate and landing distance
    """
    # Base touchdown rate calculation (more negative = harder landing)
    mean_touchdown_rate = -200 - (risk_score * 3)
    
    # Add variability based on approach parameters
    # Higher descent rate = harder landing
    descent_effect = (descent_rate - 700) * 0.3
    
    # Higher approach speed = harder touchdown rate but longer landing
    speed_effect = (approach_speed - 140) * 1.5
    
    # Proper pitch helps soften landing
    if pitch > 5:
        pitch_effect = (pitch - 5) * 10  # Flare helps reduce touchdown rate
    else:
        pitch_effect = (pitch - 5) * 20  # Insufficient flare worsens landing
    
    # Calculate final touchdown rate with some randomness
    np.random.seed(int(time.time()) % 1000)  # Change seed for variability
    randomness = np.random.normal(0, 50)  # Add noise
    
    touchdown_rate = mean_touchdown_rate - descent_effect - speed_effect + pitch_effect + randomness
    
    # Calculate landing distance (higher approach speed = longer landing)
    base_landing_distance = 2000  # Base landing distance in feet
    speed_landing_effect = (approach_speed - 140) * 50  # Each knot above/below reference adds/subtracts distance
    
    landing_distance = max(1000, base_landing_distance + speed_landing_effect)
    
    return {
        'touchdown_rate': touchdown_rate,
        'landing_distance': landing_distance
    }

def format_flight_parameter(parameter, value, optimal_range=None):
    """Format flight parameter with color-coding based on optimal range"""
    if optimal_range is None:
        return f"{parameter}: {value}"
        
    min_val, max_val = optimal_range
    
    if min_val <= value <= max_val:
        return f"{parameter}: <span style='color:green'>{value}</span>"
    elif (value < min_val and value >= min_val * 0.9) or (value > max_val and value <= max_val * 1.1):
        return f"{parameter}: <span style='color:orange'>{value}</span>"
    else:
        return f"{parameter}: <span style='color:red'>{value}</span>"
