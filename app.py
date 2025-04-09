import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time
from utils import (
    create_aircraft_visualization, 
    create_risk_gauge, 
    predict_landing_risk, 
    load_models,
    log_simulation_data
)

# Page configuration
st.set_page_config(
    page_title="Hard Landing Prediction Simulator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for warning animation
st.markdown("""
<style>
    @keyframes flashWarning {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .warning-flash {
        animation: flashWarning 1s infinite;
        background-color: #FF5757;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .simulator-header {
        background-color: #0078D7;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .parameter-container {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .parameter-label {
        font-weight: bold;
        color: #0078D7;
    }
    .normal-risk {
        color: green;
        font-weight: bold;
    }
    .medium-risk {
        color: orange;
        font-weight: bold;
    }
    .high-risk {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="simulator-header"><h1>✈️ Aircraft Hard Landing Prediction Simulator</h1></div>', unsafe_allow_html=True)
st.markdown("""
This simulation tool is designed to help pilots recognize and avoid hard landing scenarios.
Adjust flight parameters using the controls and observe how they affect landing risk in real-time.

**Learning Objectives:**
- Understand factors contributing to hard landings
- Practice approach corrections based on real-time feedback
- Develop decision-making skills for go-around situations
- Analyze parameter correlation with landing risk
""")

# Initialize session state
if 'alert_active' not in st.session_state:
    st.session_state.alert_active = False
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'current_params' not in st.session_state:
    st.session_state.current_params = {}
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 0
if 'hard_landing_prob' not in st.session_state:
    st.session_state.hard_landing_prob = 0
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'simulation_time' not in st.session_state:
    st.session_state.simulation_time = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'audio_warning' not in st.session_state:
    st.session_state.audio_warning = False
if 'simulation_count' not in st.session_state:
    st.session_state.simulation_count = 0
if 'landing_attempted' not in st.session_state:
    st.session_state.landing_attempted = False
if 'landing_results' not in st.session_state:
    st.session_state.landing_results = []

# Check if models and dataset exist, otherwise generate them
if not os.path.exists('data/flight_data.csv'):
    with st.spinner("Generating flight dataset..."):
        from dataset_generator import generate_flight_data
        generate_flight_data(5000)

if not os.path.exists('models/hard_landing_model.pkl'):
    with st.spinner("Training prediction models..."):
        from model_trainer import train_models
        train_models()

# Load models
models = load_models()

# Sidebar - Controls
st.sidebar.markdown('<div style="text-align: center;"><h2>Flight Controls</h2></div>', unsafe_allow_html=True)

# Tabs for different control categories
control_tabs = st.sidebar.tabs(["Aircraft", "Environment", "Flight Controls", "Advanced"])

# Aircraft Settings
with control_tabs[0]:
    st.subheader("Aircraft Configuration")
    aircraft_type = st.selectbox(
        "Aircraft Type",
        ["Boeing 737", "Airbus A320", "Bombardier CRJ", "Embraer E190", "Boeing 787"],
        index=0
    )
    
    flap_setting = st.selectbox(
        "Flap Setting (degrees)",
        [15, 20, 25, 30, 35, 40],
        index=3
    )
    
    gear_position = st.selectbox(
        "Gear Position",
        ["Down", "In Transit", "Up"],
        index=0
    )
    
    throttle_percentage = st.slider(
        "Throttle Percentage",
        20, 70, 40, 1,
        help="Controls engine power during approach"
    )

# Environmental Settings
with control_tabs[1]:
    st.subheader("Environmental Conditions")
    weather = st.selectbox(
        "Weather Condition",
        ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Windy", "Foggy", "Stormy"],
        index=0
    )
    
    runway_length = st.slider(
        "Runway Length (ft)",
        5000, 12000, 8000, 500,
        help="Longer runways provide more margin for landing"
    )
    
    visibility = st.slider(
        "Visibility (miles)",
        0.5, 10.0, 8.0, 0.5,
        help="Lower visibility increases landing difficulty"
    )
    
    wind_speed = st.slider(
        "Wind Speed (knots)",
        0, 35, 5, 1,
        help="Higher wind speeds make approach control more challenging"
    )
    
    wind_direction = st.slider(
        "Wind Direction (degrees)",
        0, 359, 180, 5,
        help="0° is headwind, 180° is tailwind"
    )
    
    crosswind = st.slider(
        "Crosswind Component (knots)",
        0, 25, 3, 1,
        help="Crosswind makes maintaining runway alignment more difficult"
    )

# Flight Parameters
with control_tabs[2]:
    st.subheader("Primary Flight Controls")
    approach_speed = st.slider(
        "Approach Speed (knots)",
        120, 160, 140, 1,
        help="Target speed during final approach"
    )
    
    descent_rate = st.slider(
        "Descent Rate (ft/min)",
        500, 1200, 700, 10,
        help="Vertical speed during approach (600-800 ft/min is typically ideal)"
    )
    
    altitude = st.slider(
        "Altitude (ft)",
        0, 1000, 400, 20,
        help="Current height above ground level"
    )
    
    pitch = st.slider(
        "Pitch (degrees)",
        -5.0, 15.0, 2.5, 0.5,
        help="Aircraft nose position relative to horizon (positive = nose up)"
    )
    
    roll = st.slider(
        "Roll (degrees)",
        -10.0, 10.0, 0.0, 0.5,
        help="Aircraft bank angle (0 = wings level)"
    )
    
    distance_from_threshold = st.slider(
        "Distance from Threshold (feet)",
        0, 5000, 1000, 100,
        help="Distance from runway threshold"
    )

# Advanced Parameters
with control_tabs[3]:
    st.subheader("Advanced Parameters")
    glideslope_deviation = st.slider(
        "Glideslope Deviation (dots)",
        -2.0, 2.0, 0.0, 0.1,
        help="Vertical deviation from ideal approach path (0 = on path)"
    )
    
    localizer_deviation = st.slider(
        "Localizer Deviation (dots)",
        -2.0, 2.0, 0.0, 0.1,
        help="Horizontal deviation from runway centerline (0 = on centerline)"
    )

# Simulation controls
st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Controls")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_button = st.button("Start Approach", use_container_width=True)
    attempt_landing = st.button("Attempt Landing", use_container_width=True, 
                              help="Try to land with current parameters")

with col2:
    stop_button = st.button("Pause Simulation", use_container_width=True)
    go_around = st.button("Execute Go-Around", use_container_width=True,
                        help="Abort landing and go around for another approach")

# Audio toggle for warnings
audio_enabled = st.sidebar.checkbox("Enable Audio Warnings", value=True)

# Reset button
if st.sidebar.button("Reset Simulation", use_container_width=True):
    # Reset all parameters to default values
    st.session_state.prediction_history = []
    st.session_state.simulation_time = 0
    st.session_state.start_time = None
    st.session_state.alert_active = False
    st.session_state.audio_warning = False
    st.session_state.landing_attempted = False
    st.rerun()

# Difficulty level
difficulty = st.sidebar.select_slider(
    "Difficulty Level",
    options=["Training", "Normal", "Advanced", "Expert"],
    value="Normal"
)

# Presets for different scenarios
st.sidebar.markdown("### Training Scenarios")
scenario = st.sidebar.selectbox(
    "Load Scenario",
    ["Custom", "Calm Day Perfect Approach", "Crosswind Challenge", "Low Visibility Approach", 
     "Unstable Approach", "Heavy Rain with Gusts"]
)

if scenario != "Custom" and st.sidebar.button("Load Selected Scenario"):
    # Reset simulation before loading scenario
    st.session_state.simulation_running = False
    st.session_state.alert_active = False
    st.session_state.audio_warning = False
    
    # Define preset values for different scenarios
    if scenario == "Calm Day Perfect Approach":
        # Ideal parameters for perfect approach
        weather = "Clear"
        visibility = 10.0
        wind_speed = 3
        wind_direction = 0
        crosswind = 0
        approach_speed = 140
        descent_rate = 700
        pitch = 3.0
        roll = 0.0
        glideslope_deviation = 0.0
        localizer_deviation = 0.0
        flap_setting = 30
        gear_position = "Down"
        throttle_percentage = 40
    
    elif scenario == "Crosswind Challenge":
        # Strong crosswind scenario
        weather = "Windy" 
        visibility = 8.0
        wind_speed = 20
        wind_direction = 90  # 90 degrees = direct crosswind
        crosswind = 18
        approach_speed = 145
        descent_rate = 750
        pitch = 2.5
        roll = 3.0  # Slight bank to counteract crosswind
        glideslope_deviation = 0.2
        localizer_deviation = 0.5
        flap_setting = 30
        gear_position = "Down"
        throttle_percentage = 45
    
    elif scenario == "Low Visibility Approach":
        # Poor visibility conditions
        weather = "Foggy"
        visibility = 1.5
        wind_speed = 5
        wind_direction = 30
        crosswind = 3
        approach_speed = 142
        descent_rate = 650
        pitch = 3.0
        roll = 0.5
        glideslope_deviation = 0.3
        localizer_deviation = 0.2
        flap_setting = 35
        gear_position = "Down"
        throttle_percentage = 42
    
    elif scenario == "Unstable Approach":
        # Unstable approach with multiple deviations
        weather = "Cloudy"
        visibility = 6.0
        wind_speed = 12
        wind_direction = 45
        crosswind = 8
        approach_speed = 152  # Too fast
        descent_rate = 950  # Too steep
        pitch = 1.5  # Too low
        roll = 4.0  # Excessive bank
        glideslope_deviation = 1.2  # Well above glideslope
        localizer_deviation = -0.8  # Left of centerline
        flap_setting = 25  # Insufficient flaps
        gear_position = "Down"
        throttle_percentage = 48
    
    elif scenario == "Heavy Rain with Gusts":
        # Challenging weather conditions
        weather = "Heavy Rain"
        visibility = 3.0
        wind_speed = 18
        wind_direction = 160
        crosswind = 10
        approach_speed = 148
        descent_rate = 800
        pitch = 2.8
        roll = -2.0
        glideslope_deviation = -0.5
        localizer_deviation = 0.7
        flap_setting = 35
        gear_position = "Down"
        throttle_percentage = 50
    
    # Rerun the app to update all widgets with new values
    st.rerun()

if start_button:
    st.session_state.simulation_running = True
    st.session_state.start_time = time.time()
    st.session_state.simulation_count += 1
    st.session_state.landing_attempted = False
    
if stop_button:
    st.session_state.simulation_running = False

if go_around:
    if st.session_state.simulation_running:
        # Log the go-around decision
        st.session_state.prediction_history.append({
            'timestamp': st.session_state.simulation_time,
            'risk_score': st.session_state.risk_score,
            'hard_landing_prob': st.session_state.hard_landing_prob * 100,
            'altitude': altitude,
            'approach_speed': approach_speed,
            'descent_rate': descent_rate,
            'pitch': pitch,
            'roll': roll,
            'glideslope_deviation': glideslope_deviation,
            'localizer_deviation': localizer_deviation,
            'event': 'go_around'
        })
        
        st.success("Go-around executed successfully. Good decision making!")
        
        # Reset altitude to simulate climb
        altitude = 800
        
        # Update session state to show go-around was performed
        st.session_state.landing_results.append({
            'scenario': scenario if scenario != "Custom" else "Custom Approach",
            'result': "Go-Around",
            'risk_score': st.session_state.risk_score,
            'time': time.strftime('%M:%S', time.gmtime(st.session_state.simulation_time))
        })
        
        # Pause simulation briefly
        st.session_state.simulation_running = False
        st.session_state.alert_active = False
        
if attempt_landing:
    if st.session_state.simulation_running and altitude < 100:
        # Mark that landing was attempted
        st.session_state.landing_attempted = True
        
        # Calculate landing success based on risk score
        landing_risk = st.session_state.risk_score
        landing_success = np.random.random() * 100 > landing_risk
        
        # Calculate landing rate (lower = harder landing)
        base_landing_rate = -300 - (landing_risk * 3)
        noise = np.random.normal(0, 50)
        landing_rate = base_landing_rate + noise
        
        # Log the landing attempt
        landing_event = {
            'timestamp': st.session_state.simulation_time,
            'risk_score': landing_risk,
            'hard_landing_prob': st.session_state.hard_landing_prob * 100,
            'altitude': altitude,
            'approach_speed': approach_speed,
            'descent_rate': descent_rate,
            'pitch': pitch,
            'roll': roll,
            'glideslope_deviation': glideslope_deviation,
            'localizer_deviation': localizer_deviation,
            'landing_rate': landing_rate,
            'landing_success': landing_success,
            'event': 'landing_attempt'
        }
        
        st.session_state.prediction_history.append(landing_event)
        
        # Update landing results history
        if landing_success:
            result = "Success" if landing_rate > -400 else "Hard Landing"
            result_color = "green" if landing_rate > -400 else "orange"
        else:
            result = "Landing Accident"
            result_color = "red"
            
        st.session_state.landing_results.append({
            'scenario': scenario if scenario != "Custom" else "Custom Approach",
            'result': result,
            'landing_rate': landing_rate,
            'risk_score': landing_risk,
            'time': time.strftime('%M:%S', time.gmtime(st.session_state.simulation_time)),
            'color': result_color
        })
        
        # Stop simulation after landing attempt
        st.session_state.simulation_running = False
        st.session_state.alert_active = False
        
        # Show landing result message
        if landing_success:
            if landing_rate > -400:
                st.success(f"Successful landing! Touchdown rate: {landing_rate:.0f} ft/min")
            else:
                st.warning(f"Hard landing detected. Touchdown rate: {landing_rate:.0f} ft/min")
        else:
            st.error(f"Landing accident! Excessive touchdown rate: {landing_rate:.0f} ft/min")
            
    elif st.session_state.simulation_running and altitude >= 100:
        st.warning("Too high for landing. Descend below 100 feet to attempt landing.")
    else:
        st.info("Start the simulation before attempting to land.")

# Main display layout
col1, col2 = st.columns([3, 1])

with col1:
    # Aircraft visualization
    st.subheader("Aircraft Approach Visualization")
    
    # Create input data dictionary for prediction
    input_data = {
        'aircraft_type': aircraft_type,
        'weather': weather,
        'runway_length_ft': runway_length,
        'visibility_miles': visibility,
        'wind_speed_knots': wind_speed,
        'wind_direction_degrees': wind_direction,
        'crosswind_component_knots': crosswind,
        'approach_speed_knots': approach_speed,
        'descent_rate_fpm': descent_rate,
        'altitude_ft': altitude,
        'glideslope_deviation_dots': glideslope_deviation,
        'localizer_deviation_dots': localizer_deviation,
        'pitch_degrees': pitch,
        'roll_degrees': roll,
        'flap_setting_degrees': flap_setting,
        'gear_position': gear_position,
        'distance_from_threshold_feet': distance_from_threshold,
        'throttle_percentage': throttle_percentage
    }
    
    # Make prediction when simulation is running
    if st.session_state.simulation_running:
        # Update simulation time
        current_time = time.time()
        if st.session_state.start_time:
            st.session_state.simulation_time = current_time - st.session_state.start_time
        
        st.session_state.current_params = input_data
        st.session_state.hard_landing_prob, st.session_state.risk_score = predict_landing_risk(input_data)
        
        # Log prediction
        timestamp = st.session_state.simulation_time
        prediction_data = {
            'timestamp': timestamp,
            'risk_score': st.session_state.risk_score,
            'hard_landing_prob': st.session_state.hard_landing_prob * 100,
            'altitude': altitude,
            'approach_speed': approach_speed,
            'descent_rate': descent_rate,
            'pitch': pitch,
            'roll': roll,
            'glideslope_deviation': glideslope_deviation,
            'localizer_deviation': localizer_deviation
        }
        
        st.session_state.prediction_history.append(prediction_data)
        
        # Limit history length
        if len(st.session_state.prediction_history) > 200:
            st.session_state.prediction_history = st.session_state.prediction_history[-200:]
        
        # Activate alert if risk is high
        if st.session_state.risk_score >= 70 and not st.session_state.alert_active:
            st.session_state.alert_active = True
            st.session_state.audio_warning = True
            
        # Deactivate alert if risk is low
        if st.session_state.risk_score < 50 and st.session_state.alert_active:
            st.session_state.alert_active = False
            st.session_state.audio_warning = False
            
        # Log the simulation data for later analysis
        log_simulation_data(input_data, (st.session_state.hard_landing_prob, st.session_state.risk_score))
    
    # Create and display the aircraft visualization
    fig = create_aircraft_visualization(
        pitch, 
        roll, 
        altitude, 
        st.session_state.risk_score if st.session_state.simulation_running else 0,
        approach_speed,
        descent_rate,
        st.session_state.get('wind_speed', 0),
        st.session_state.get('wind_direction', 0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Audio warning (using HTML5 audio)
    if st.session_state.audio_warning and audio_enabled:
        st.markdown("""
        <audio autoplay loop>
            <source src="data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAA==" type="audio/wav">
        </audio>
        <script>
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            oscillator.type = 'sawtooth';
            oscillator.frequency.value = 440;
            gainNode.gain.value = 0.3;
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            oscillator.start();
            
            // Create beeping effect
            setInterval(() => {
                gainNode.gain.value = gainNode.gain.value > 0 ? 0 : 0.3;
            }, 500);
        </script>
        """, unsafe_allow_html=True)
    
    # Alert indicator
    if st.session_state.alert_active:
        st.markdown('<div class="warning-flash"><h3>⚠️ WARNING: HIGH RISK OF HARD LANDING DETECTED!</h3><h4>INITIATE GO-AROUND OR CORRECT APPROACH IMMEDIATELY!</h4></div>', unsafe_allow_html=True)
        
        # Corrective action suggestions based on parameters
        suggestions = []
        
        if descent_rate > 800:
            suggestions.append("• **Reduce descent rate** below 800 ft/min (Currently: {:.0f} ft/min)".format(descent_rate))
            
        if abs(glideslope_deviation) > 1:
            suggestions.append("• **Correct glideslope deviation** (Currently: {:.1f} dots)".format(glideslope_deviation))
            
        if abs(localizer_deviation) > 1:
            suggestions.append("• **Correct localizer deviation** (Currently: {:.1f} dots)".format(localizer_deviation))
            
        if approach_speed > 150:
            suggestions.append("• **Reduce approach speed** (Currently: {:.0f} knots)".format(approach_speed))
            
        if approach_speed < 130:
            suggestions.append("• **Increase approach speed** (Currently: {:.0f} knots)".format(approach_speed))
            
        if abs(roll) > 5:
            suggestions.append("• **Level wings** (Current roll: {:.1f}°)".format(roll))
            
        if pitch < 0:
            suggestions.append("• **Increase pitch attitude** (Current pitch: {:.1f}°)".format(pitch))
            
        if crosswind > 15:
            suggestions.append("• **Apply proper crosswind correction** (Crosswind: {:.0f} knots)".format(crosswind))
            
        if gear_position != "Down" and altitude < 500:
            suggestions.append("• **Lower landing gear** (Currently: {})".format(gear_position))
            
        st.info("### Corrective Actions:\n" + "\n".join(suggestions) if suggestions else "### Recommended Action: **Initiate go-around procedure**")

with col2:
    # Risk gauge
    st.subheader("Landing Risk Assessment")
    risk_gauge = create_risk_gauge(st.session_state.risk_score if st.session_state.simulation_running else 0)
    st.plotly_chart(risk_gauge, use_container_width=True)
    
    # Hard landing probability
    landing_prob = st.session_state.hard_landing_prob * 100 if st.session_state.simulation_running else 0
    
    # Determine color based on probability value
    if landing_prob < 30:
        prob_color = "normal-risk"
    elif landing_prob < 70:
        prob_color = "medium-risk"
    else:
        prob_color = "high-risk"
    
    st.markdown(f"<div class='parameter-container'><span class='parameter-label'>Hard Landing Probability:</span> <span class='{prob_color}'>{landing_prob:.1f}%</span></div>", unsafe_allow_html=True)
    
    # Simulation time
    formatted_time = time.strftime('%M:%S', time.gmtime(st.session_state.simulation_time))
    st.markdown(f"<div class='parameter-container'><span class='parameter-label'>Simulation Time:</span> {formatted_time}</div>", unsafe_allow_html=True)
    
    # Simulation status
    status = "Running" if st.session_state.simulation_running else "Paused"
    status_color = "normal-risk" if st.session_state.simulation_running else ""
    st.markdown(f"<div class='parameter-container'><span class='parameter-label'>Status:</span> <span class='{status_color}'>{status}</span></div>", unsafe_allow_html=True)
    
    # Create key metrics card
    st.markdown("### Key Flight Parameters")
    
    # Use columns for metrics
    params_col1, params_col2 = st.columns(2)
    
    with params_col1:
        st.metric("Descent Rate", f"{descent_rate} ft/min", 
                 delta=f"{descent_rate-700}" if descent_rate != 700 else None,
                 delta_color="inverse")
        st.metric("Altitude", f"{altitude} ft")
        st.metric("Pitch", f"{pitch}°")
    
    with params_col2:
        st.metric("Approach Speed", f"{approach_speed} kts",
                 delta=f"{approach_speed-140}" if approach_speed != 140 else None,
                 delta_color="inverse")
        st.metric("Dist. to Threshold", f"{distance_from_threshold} ft")
        st.metric("Roll", f"{roll}°", 
                 delta=None if abs(roll) < 0.5 else f"{roll}°",
                 delta_color="inverse")
    
    # Glideslope and localizer deviations
    gs_color = "inverse" if abs(glideslope_deviation) > 0.5 else "normal"
    loc_color = "inverse" if abs(localizer_deviation) > 0.5 else "normal"
    
    st.metric("Glideslope", f"{glideslope_deviation:+.1f} dots", 
             delta=None if abs(glideslope_deviation) < 0.1 else f"{glideslope_deviation:+.1f}",
             delta_color=gs_color)
    
    st.metric("Localizer", f"{localizer_deviation:+.1f} dots", 
             delta=None if abs(localizer_deviation) < 0.1 else f"{localizer_deviation:+.1f}",
             delta_color=loc_color)
    
    # Weather indicator with icons
    weather_icons = {
        "Clear": "☀️",
        "Cloudy": "☁️",
        "Light Rain": "🌦️",
        "Heavy Rain": "🌧️",
        "Windy": "💨",
        "Foggy": "🌫️",
        "Stormy": "⛈️"
    }
    
    # Wind component visualization
    st.markdown("### Wind Components")
    
    # Calculate headwind/tailwind component
    wind_rad = np.radians(wind_direction)
    headwind = wind_speed * np.cos(wind_rad)
    crosswind_comp = wind_speed * np.sin(wind_rad)
    
    # Display wind information
    wind_col1, wind_col2 = st.columns(2)
    
    with wind_col1:
        st.metric("Headwind", f"{abs(headwind):.1f} kts", 
                 delta="Headwind" if headwind > 0 else "Tailwind",
                 delta_color="normal" if headwind > 0 else "inverse")
    
    with wind_col2:
        st.metric("Crosswind", f"{abs(crosswind_comp):.1f} kts", 
                 delta="Right" if crosswind_comp > 0 else "Left",
                 delta_color="normal")
    
    st.markdown(f"**Weather:** {weather_icons.get(weather, '☀️')} {weather}")
    st.markdown(f"**Visibility:** {visibility} miles")

    # Landing history/results
    if st.session_state.landing_results:
        st.markdown("### Landing Results")
        landing_df = pd.DataFrame(st.session_state.landing_results)
        
        # Custom styling for the results table
        html_table = "<div style='max-height: 200px; overflow-y: auto;'>"
        html_table += "<table style='width: 100%;'>"
        html_table += "<tr><th>Scenario</th><th>Result</th><th>Risk</th><th>Time</th></tr>"
        
        for i, row in enumerate(st.session_state.landing_results):
            result_color = row.get('color', 'black')
            html_table += f"<tr>"
            html_table += f"<td>{row['scenario']}</td>"
            html_table += f"<td style='color: {result_color};'>{row['result']}</td>"
            html_table += f"<td>{row['risk_score']:.1f}%</td>"
            html_table += f"<td>{row['time']}</td>"
            html_table += f"</tr>"
        
        html_table += "</table></div>"
        
        st.markdown(html_table, unsafe_allow_html=True)

# Historical data graph
if st.session_state.prediction_history:
    st.subheader("Flight Data Analysis")
    
    # Convert prediction history to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    # Create tabs for different analyses
    analysis_tabs = st.tabs(["Risk Trend", "Parameter Correlation", "Flight Path Analysis"])
    
    with analysis_tabs[0]:
        # Time series of risk score
        st.subheader("Risk Score Trend")
        
        # Create line chart of risk score over time
        fig = px.line(history_df, x='timestamp', y='risk_score', 
                    title='Risk Score History',
                    labels={'timestamp': 'Time (seconds)', 'risk_score': 'Risk Score (%)'})
        
        # Add danger threshold line
        fig.add_shape(
            type="line",
            x0=history_df['timestamp'].min(),
            y0=70,
            x1=history_df['timestamp'].max(),
            y1=70,
            line=dict(color="red", width=2, dash="dash"),
            name="Danger Threshold"
        )
        
        # Add warning threshold line
        fig.add_shape(
            type="line",
            x0=history_df['timestamp'].min(),
            y0=30,
            x1=history_df['timestamp'].max(),
            y1=30,
            line=dict(color="orange", width=2, dash="dash"),
            name="Warning Threshold"
        )
        
        # Mark events on the graph if they exist
        if 'event' in history_df.columns:
            events_df = history_df[history_df['event'].notna()]
            
            for event_type in events_df['event'].unique():
                event_data = events_df[events_df['event'] == event_type]
                
                # Different markers for different events
                if event_type == 'go_around':
                    fig.add_trace(go.Scatter(
                        x=event_data['timestamp'],
                        y=event_data['risk_score'],
                        mode='markers',
                        marker=dict(symbol='star', size=12, color='yellow'),
                        name='Go Around',
                        hoverinfo='text',
                        text='Go Around Decision'
                    ))
                elif event_type == 'landing_attempt':
                    fig.add_trace(go.Scatter(
                        x=event_data['timestamp'],
                        y=event_data['risk_score'],
                        mode='markers',
                        marker=dict(symbol='circle', size=12, color='green'),
                        name='Landing Attempt',
                        hoverinfo='text',
                        text=['Landing: ' + ('Success' if s else 'Accident') for s in event_data['landing_success']]
                    ))
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Simulation Time (seconds)",
            yaxis_title="Risk Score (%)",
            yaxis_range=[0, 100],
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Multiple parameters over time
        st.subheader("Flight Parameters Over Time")
        
        # Select parameters to display
        selected_params = st.multiselect(
            "Select parameters to display",
            ['altitude', 'approach_speed', 'descent_rate', 'pitch', 'roll', 
             'glideslope_deviation', 'localizer_deviation'],
            default=['altitude', 'descent_rate']
        )
        
        if selected_params:
            # Normalize parameters for comparison
            norm_df = history_df.copy()
            for param in selected_params:
                if param in norm_df.columns:
                    max_val = norm_df[param].max()
                    min_val = norm_df[param].min()
                    if max_val > min_val:
                        norm_df[f"{param}_norm"] = (norm_df[param] - min_val) / (max_val - min_val) * 100
                    else:
                        norm_df[f"{param}_norm"] = 50  # Default if no variation
            
            # Create multi-line chart
            fig = go.Figure()
            
            # Add line for each parameter
            for param in selected_params:
                if param in norm_df.columns:
                    fig.add_trace(go.Scatter(
                        x=norm_df['timestamp'],
                        y=norm_df[f"{param}_norm"],
                        mode='lines',
                        name=param
                    ))
            
            # Add risk score
            fig.add_trace(go.Scatter(
                x=norm_df['timestamp'],
                y=norm_df['risk_score'],
                mode='lines',
                name='risk_score',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Normalized Parameter Values Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Normalized Value (%)",
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with analysis_tabs[1]:
        st.subheader("Parameter Correlation with Risk")
        
        # Create more detailed correlation analysis
        correlation_params = ['descent_rate', 'approach_speed', 'altitude', 
                              'pitch', 'roll', 'glideslope_deviation', 'localizer_deviation']
        
        # Let user select parameter to analyze
        param_to_analyze = st.selectbox(
            "Select parameter to analyze against risk score:", 
            correlation_params,
            index=0
        )
        
        if param_to_analyze in history_df.columns:
            # Create scatter plot
            fig = px.scatter(history_df, x=param_to_analyze, y='risk_score',
                            color='risk_score',
                            color_continuous_scale=['green', 'yellow', 'red'],
                            labels={param_to_analyze: f'{param_to_analyze.replace("_", " ").title()}', 
                                    'risk_score': 'Risk Score (%)'},
                            title=f'{param_to_analyze.replace("_", " ").title()} Impact on Landing Risk')
            
            # Add trendline
            fig.update_traces(marker=dict(size=8))
            
            # Add best fit line
            if len(history_df) > 2:
                fig.add_traces(
                    px.scatter(history_df, x=param_to_analyze, y='risk_score', trendline='ols')
                    .data[1]
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation coefficient
            if len(history_df) > 2:
                corr = history_df[[param_to_analyze, 'risk_score']].corr().iloc[0,1]
                st.info(f"Correlation coefficient: {corr:.2f}")
                
                # Interpretation of correlation
                if abs(corr) < 0.3:
                    st.write("This parameter has a weak relationship with landing risk in this simulation.")
                elif abs(corr) < 0.7:
                    st.write("This parameter has a moderate relationship with landing risk.")
                else:
                    st.write("This parameter has a strong relationship with landing risk.")
                
                # Provide educational guidance based on parameter
                if param_to_analyze == 'descent_rate':
                    st.markdown("""
                    **Impact of Descent Rate:**
                    - Ideal descent rate is typically 600-800 ft/min for most commercial aircraft
                    - Higher descent rates increase the risk of hard landings and require more precise timing for flare
                    - Excessive descent rates require larger control inputs to arrest descent, increasing pilot workload
                    """)
                elif param_to_analyze == 'approach_speed':
                    st.markdown("""
                    **Impact of Approach Speed:**
                    - Approach speed should be maintained within ±5 knots of reference speed (Vref)
                    - Higher speeds result in longer flare, increased floating and runway usage
                    - Lower speeds reduce control effectiveness and increase stall risk
                    - Each additional 5 knots above Vref adds approximately 150 feet to landing distance
                    """) 
                elif param_to_analyze == 'glideslope_deviation':
                    st.markdown("""
                    **Impact of Glideslope Deviation:**
                    - Staying on the glideslope ensures proper positioning for landing
                    - Above glideslope: results in shallow approach and potential for floating/long landing
                    - Below glideslope: steeper approach increases descent rate and hard landing risk
                    - Deviations require power and pitch corrections; significant deviations may warrant a go-around
                    """)
    
    with analysis_tabs[2]:
        st.subheader("3D Flight Path Visualization")
        
        if len(history_df) > 5:
            # Create a 3D scatter plot of the aircraft's path
            fig = go.Figure(data=[go.Scatter3d(
                x=history_df['timestamp'],
                y=history_df['localizer_deviation'],
                z=history_df['altitude'],
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=history_df['risk_score'],
                    colorscale='Viridis',
                    colorbar=dict(title="Risk Score"),
                    opacity=0.8
                ),
                line=dict(
                    color='darkblue',
                    width=2
                )
            )])
            
            # Add ideal approach path as reference
            if history_df['timestamp'].max() > 0:
                # Create an idealized descent profile
                ideal_time = np.linspace(0, history_df['timestamp'].max(), 50)
                max_alt = history_df['altitude'].max()
                ideal_alt = max_alt - (ideal_time / ideal_time.max()) * max_alt
                
                fig.add_trace(go.Scatter3d(
                    x=ideal_time,
                    y=np.zeros(len(ideal_time)),  # Centered on runway
                    z=ideal_alt,
                    mode='lines',
                    line=dict(color='green', width=3, dash='dash'),
                    name='Ideal Path'
                ))
            
            fig.update_layout(
                title="3D Flight Path Visualization",
                scene=dict(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Lateral Position (localizer deviation)",
                    zaxis_title="Altitude (feet)",
                    aspectratio=dict(x=1.5, y=1, z=1)
                ),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("This 3D visualization shows your aircraft's trajectory. Points are colored by risk score - redder points indicate higher risk of hard landing.")

# Performance Metrics
if st.session_state.landing_results:
    with st.expander("Performance Summary", expanded=False):
        st.markdown("### Your Performance Metrics")
        
        # Calculate statistics
        total_approaches = len(st.session_state.landing_results)
        successful_landings = sum(1 for r in st.session_state.landing_results if r['result'] == "Success")
        hard_landings = sum(1 for r in st.session_state.landing_results if r['result'] == "Hard Landing")
        accidents = sum(1 for r in st.session_state.landing_results if r['result'] == "Landing Accident")
        go_arounds = sum(1 for r in st.session_state.landing_results if r['result'] == "Go-Around")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Approaches", total_approaches)
            st.metric("Successful Landings", successful_landings)
        
        with col2:
            st.metric("Hard Landings", hard_landings)
            st.metric("Go-Arounds", go_arounds)
        
        with col3:
            st.metric("Landing Accidents", accidents)
            success_rate = successful_landings / total_approaches * 100 if total_approaches > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        # Display feedback based on performance
        if total_approaches >= 3:
            st.markdown("### Feedback")
            if success_rate >= 80:
                st.success("Excellent performance! You're demonstrating good judgment and landing technique.")
            elif success_rate >= 60:
                st.info("Good performance. Continue practicing to improve consistency in challenging conditions.")
            else:
                st.warning("You're experiencing difficulty with landings. Focus on maintaining stable approaches and recognizing when to execute a go-around.")
                
            # Specific recommendations
            if hard_landings > successful_landings:
                st.markdown("**Focus Area**: Work on proper flare technique and descent rate management during final approach.")
            if accidents > 0:
                st.markdown("**Safety Critical**: Review go-around decision making. Remember that executing a go-around is always better than risking an unsafe landing.")
            if go_arounds > total_approaches / 2:
                st.markdown("**Confidence Building**: While go-arounds show good judgment, work on approach stability to reduce the need for them.")

# Training information section
with st.expander("Flight Training Information"):
    st.markdown("""
    ## Hard Landing Prevention Guidelines
    
    ### Factors Contributing to Hard Landings
    
    #### 1. Approach Stability
    - **Unstable approach**: Excessive adjustments in pitch, power, or descent rate
    - **Glideslope deviations**: Not maintaining proper approach angle
    - **Airspeed control**: Too fast or too slow approach speeds
    
    #### 2. Environmental Factors
    - **Crosswinds**: Improper crosswind correction technique
    - **Wind shear**: Sudden changes in wind direction or speed
    - **Runway conditions**: Wet, icy, or contaminated surfaces
    - **Visibility**: Reduced visual references in poor weather
    
    #### 3. Aircraft Configuration
    - **Flap settings**: Improper flap configuration for conditions
    - **Landing gear**: Late gear extension
    - **Weight and balance**: Forward or aft CG can affect flare characteristics
    
    ### Technique Improvement
    
    #### Proper Approach Technique
    - Maintain stable approach with proper power and pitch
    - Target descent rate between 600-800 ft/min
    - Keep airspeed within ±5 knots of reference speed
    - Maintain wings level (minimal bank angle)
    - Stay within 1 dot of glideslope and localizer
    
    #### Flare Technique
    - Begin flare at appropriate height (typically 20-30 ft)
    - Gradually reduce descent rate to near zero at touchdown
    - Maintain directional control throughout flare
    - Control sink rate with proper pitch adjustment
    - Avoid floating or excessive flare
    
    #### Go-Around Decision Making
    - **When to go around**:
        - Unstable approach below 500 ft
        - Excessive deviations from glideslope/localizer
        - Abnormal sink rate or airspeed
        - When risk assessment shows high probability of hard landing
        - Any time you feel uncomfortable with the approach
    """)

# Footer
st.markdown("---")
st.markdown("**Hard Landing Prediction Simulator** | Pilot Training Tool | v1.0")
