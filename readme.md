# Hard Landing Prediction Simulator

This application provides a realistic simulation environment for pilot training, specifically focusing on predicting and preventing hard landings during the approach phase of commercial flights.

## Overview

The Hard Landing Prediction Simulator integrates machine learning with an interactive flight simulator to predict the risk of hard landings in real-time. Pilots can use this tool to:

1. Practice approach and landing techniques in various environmental conditions
2. Receive real-time feedback on landing risk
3. Understand the key factors contributing to hard landings
4. Develop better decision-making skills for go-around situations
5. Analyze flight parameter correlations with landing risk

## Features

- Interactive flight simulation with aircraft orientation visualization
- Real-time hard landing risk assessment using machine learning
- Customizable environmental conditions (weather, wind, visibility)
- Flight parameter controls (approach speed, descent rate, glideslope, etc.)
- Visual and audio alerts for high-risk situations
- Detailed flight data analysis with parameter correlation
- 3D flight path visualization
- Training scenarios for different challenging conditions
- Performance tracking and feedback

## Getting Started

1. Run the data generator to create the flight dataset:
   ```
   python dataset_generator.py
   ```

2. Train the machine learning models:
   ```
   python model_trainer.py
   ```

3. Launch the application:
   ```
   streamlit run app.py
   ```

## Using the Simulator

### Controls

The simulator provides controls for:

- **Aircraft Configuration**: Aircraft type, flap settings, gear position, throttle
- **Environmental Conditions**: Weather, runway length, visibility, wind components
- **Flight Controls**: Approach speed, descent rate, altitude, pitch, roll, etc.
- **Advanced Parameters**: Glideslope and localizer deviations

### Flight Process

1. Select your desired scenario or customize parameters
2. Click "Start Approach" to begin the simulation
3. Adjust parameters during the approach to maintain a stable flight path
4. If risk increases, make corrections or execute a go-around
5. When at an appropriate altitude, attempt landing
6. Review the landing results and flight analysis
7. Reset and try again to improve technique

### Training Scenarios

The simulator includes pre-configured scenarios to practice different challenging situations:

- Calm Day Perfect Approach
- Crosswind Challenge
- Low Visibility Approach
- Unstable Approach
- Heavy Rain with Gusts

## Technical Details

### Machine Learning Model

The system uses two machine learning models:

1. **Classification Model**: Predicts the probability of a hard landing (binary outcome)
2. **Regression Model**: Predicts the risk score on a continuous scale (0-100%)

### Dataset

The training dataset consists of 5,000 simulated approaches with the following key features:
- Aircraft parameters (type, configuration)
- Environmental conditions (weather, visibility, runway)
- Flight parameters (speed, descent rate, altitude)
- Approach stability indicators (glideslope/localizer deviations)

### Development

The application is built with:
- Python 3.8+
- Streamlit for the interactive web interface
- Pandas and NumPy for data processing
- Scikit-learn for machine learning models
- Plotly for interactive visualizations

## Educational Resources

The simulator includes comprehensive educational resources on:
- Factors contributing to hard landings
- Proper approach and flare techniques
- Go-around decision making
- Parameter correlation analysis

## License

This project is for educational purposes only.

## Acknowledgements

This simulator was developed as a training tool for pilots to enhance safety during the approach and landing phases of flight.
