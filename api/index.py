"""
CardioML - Heart Disease Prediction Web Application (Vercel-optimized)
Developed by: Zayyan

This version uses embedded model coefficients to avoid large dependencies.
"""

from flask import Flask, render_template, request, jsonify
import json
import math
import os

app = Flask(__name__)

# Embedded model parameters (extracted from trained LogisticRegression)
# These are the coefficients and intercept from the trained model
MODEL_PARAMS = {
    "features": ["heart rate", "RDW", "Leucocyte", "PT", "INR", "Urea nitrogen", "Blood potassium", "Anion gap", "Lactic acid"],
    "coefficients": [0.3347186776471683, 0.07779145517967605, 0.23986260792477876, -0.08401034026743019, 0.3674307488917965, 0.5286966766093179, 0.08499715742116515, 0.18926992991158484, 0.4023498027729396],
    "intercept": -2.0222023633036597,
    "scaler_mean": [85.0424831053271, 16.024249780607, 10.822823208142523, 17.400868426799068, 1.6194814926845795, 37.62586241527102, 4.176773328011682, 14.116525133983645, 1.9003470912009346],
    "scaler_std": [16.24265201806411, 2.1123762752399253, 5.460986297347533, 7.449013462017955, 0.8574222582571968, 21.55397772155045, 0.3856394100290199, 2.6411958493521253, 0.9967583125362679]
}

selected_features = MODEL_PARAMS["features"]

def sigmoid(x):
    """Sigmoid function for logistic regression"""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)

def predict(input_values):
    """Make prediction using embedded model parameters"""
    # Scale the input
    scaled_input = []
    for i, val in enumerate(input_values):
        scaled_val = (val - MODEL_PARAMS["scaler_mean"][i]) / MODEL_PARAMS["scaler_std"][i]
        scaled_input.append(scaled_val)
    
    # Calculate linear combination
    z = MODEL_PARAMS["intercept"]
    for i, coef in enumerate(MODEL_PARAMS["coefficients"]):
        z += coef * scaled_input[i]
    
    # Apply sigmoid to get probability
    prob_disease = sigmoid(z)
    prob_no_disease = 1 - prob_disease
    
    # Make prediction
    prediction = 1 if prob_disease >= 0.5 else 0
    
    return {
        "prediction": prediction,
        "probability_no_disease": round(prob_no_disease * 100, 2),
        "probability_disease": round(prob_disease * 100, 2),
        "risk_level": "High Risk" if prediction == 1 else "Low Risk"
    }

# Feature information for the form
FEATURE_INFO = {
    'heart rate': {'label': 'Heart Rate', 'type': 'number', 'min': 30, 'max': 200, 'default': 75, 'unit': 'bpm'},
    'RDW': {'label': 'RDW', 'type': 'number', 'min': 10, 'max': 25, 'default': 13, 'step': 0.1, 'unit': '%'},
    'Leucocyte': {'label': 'Leucocyte Count', 'type': 'number', 'min': 1, 'max': 30, 'default': 7, 'step': 0.1, 'unit': 'K/µL'},
    'PT': {'label': 'Prothrombin Time', 'type': 'number', 'min': 8, 'max': 40, 'default': 12, 'step': 0.1, 'unit': 's'},
    'INR': {'label': 'INR', 'type': 'number', 'min': 0.5, 'max': 5, 'default': 1, 'step': 0.1, 'unit': ''},
    'Urea nitrogen': {'label': 'Urea Nitrogen', 'type': 'number', 'min': 5, 'max': 150, 'default': 15, 'unit': 'mg/dL'},
    'Blood potassium': {'label': 'Blood Potassium', 'type': 'number', 'min': 2, 'max': 8, 'default': 4.2, 'step': 0.1, 'unit': 'mEq/L'},
    'Anion gap': {'label': 'Anion Gap', 'type': 'number', 'min': 3, 'max': 30, 'default': 12, 'unit': 'mEq/L'},
    'Lactic acid': {'label': 'Lactic Acid', 'type': 'number', 'min': 0.3, 'max': 10, 'default': 1, 'step': 0.1, 'unit': 'mmol/L'},
}

@app.route('/')
def home():
    """Render the home page with prediction form"""
    features_for_form = []
    for feat in selected_features:
        if feat in FEATURE_INFO:
            info = FEATURE_INFO[feat].copy()
            info['name'] = feat
            features_for_form.append(info)
        else:
            features_for_form.append({
                'name': feat,
                'label': feat.replace('_', ' ').title(),
                'type': 'number',
                'default': 0,
                'unit': ''
            })
    return render_template('index.html', features=features_for_form)

@app.route('/predict', methods=['POST'])
def predict_route():
    """Make prediction based on input features"""
    try:
        # Get input values
        input_data = []
        for feat in selected_features:
            value = request.form.get(feat, 0)
            try:
                input_data.append(float(value))
            except ValueError:
                input_data.append(0)
        
        # Make prediction
        result = predict(input_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

# For Vercel
app = app

if __name__ == '__main__':
    print("🫀 CardioML - Heart Disease Prediction")
    print("=" * 40)
    print(f"Selected features: {selected_features}")
    print("=" * 40)
    print("Starting web server...")
    app.run(debug=True, port=5000)
