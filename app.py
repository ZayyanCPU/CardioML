"""
CardioML - Heart Disease Prediction Web Application
Developed by: Zayyan
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
import os

app = Flask(__name__)

# Global variables for model and preprocessing
model = None
scaler = None
imputer = None
selected_features = None

def train_and_save_model():
    """Train the Logistic Regression model and save it"""
    global model, scaler, imputer, selected_features
    
    # Load data
    data = pd.read_csv('Heart Disease.csv')
    
    # Define features and target
    target = 'outcome (Target)'
    features = data.columns.drop(['outcome (Target)', 'group', 'ID'])
    
    # Calculate correlations and select high correlation features
    matrix = data.corr()
    correlations = matrix[target]
    highCorr = correlations[correlations > 0.129].index
    highCorr = [f for f in highCorr if f not in ['outcome (Target)', 'group', 'ID']]
    
    selected_features = highCorr
    
    # Prepare data
    dataset = data.dropna()
    X = dataset[selected_features]
    y = dataset[target]
    
    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Logistic Regression with best parameters
    model = LogisticRegression(C=1, solver='liblinear', max_iter=300)
    model.fit(X_train, y_train)
    
    # Save model, scaler, imputer, and features
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'imputer': imputer,
            'features': selected_features
        }, f)
    
    print(f"Model trained! Test accuracy: {model.score(X_test, y_test):.4f}")
    return model, scaler, imputer, selected_features

def load_model():
    """Load the trained model or train a new one"""
    global model, scaler, imputer, selected_features
    
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data['scaler']
            imputer = data['imputer']
            selected_features = data['features']
    else:
        model, scaler, imputer, selected_features = train_and_save_model()

# Feature information for the form
FEATURE_INFO = {
    'age': {'label': 'Age', 'type': 'number', 'min': 0, 'max': 120, 'default': 50, 'unit': 'years'},
    'BMI': {'label': 'BMI', 'type': 'number', 'min': 10, 'max': 60, 'default': 25, 'step': 0.1, 'unit': 'kg/m²'},
    'heart rate': {'label': 'Heart Rate', 'type': 'number', 'min': 30, 'max': 200, 'default': 75, 'unit': 'bpm'},
    'Systolic blood pressure': {'label': 'Systolic BP', 'type': 'number', 'min': 70, 'max': 250, 'default': 120, 'unit': 'mmHg'},
    'Diastolic blood pressure': {'label': 'Diastolic BP', 'type': 'number', 'min': 40, 'max': 150, 'default': 80, 'unit': 'mmHg'},
    'Respiratory rate': {'label': 'Respiratory Rate', 'type': 'number', 'min': 8, 'max': 40, 'default': 16, 'unit': '/min'},
    'temperature': {'label': 'Temperature', 'type': 'number', 'min': 34, 'max': 42, 'default': 36.6, 'step': 0.1, 'unit': '°C'},
    'SP O2': {'label': 'SpO2', 'type': 'number', 'min': 70, 'max': 100, 'default': 98, 'unit': '%'},
    'Urine output': {'label': 'Urine Output', 'type': 'number', 'min': 0, 'max': 10000, 'default': 1500, 'unit': 'mL/day'},
    'hematocrit': {'label': 'Hematocrit', 'type': 'number', 'min': 15, 'max': 60, 'default': 40, 'step': 0.1, 'unit': '%'},
    'RBC': {'label': 'RBC Count', 'type': 'number', 'min': 1, 'max': 8, 'default': 4.5, 'step': 0.01, 'unit': 'M/µL'},
    'MCH': {'label': 'MCH', 'type': 'number', 'min': 20, 'max': 40, 'default': 29, 'step': 0.1, 'unit': 'pg'},
    'MCHC': {'label': 'MCHC', 'type': 'number', 'min': 25, 'max': 40, 'default': 33, 'step': 0.1, 'unit': 'g/dL'},
    'MCV': {'label': 'MCV', 'type': 'number', 'min': 60, 'max': 120, 'default': 90, 'step': 0.1, 'unit': 'fL'},
    'RDW': {'label': 'RDW', 'type': 'number', 'min': 10, 'max': 25, 'default': 13, 'step': 0.1, 'unit': '%'},
    'Leucocyte': {'label': 'Leucocyte Count', 'type': 'number', 'min': 1, 'max': 30, 'default': 7, 'step': 0.1, 'unit': 'K/µL'},
    'Platelets': {'label': 'Platelet Count', 'type': 'number', 'min': 50, 'max': 600, 'default': 250, 'unit': 'K/µL'},
    'Neutrophils': {'label': 'Neutrophils', 'type': 'number', 'min': 20, 'max': 95, 'default': 60, 'step': 0.1, 'unit': '%'},
    'Basophils': {'label': 'Basophils', 'type': 'number', 'min': 0, 'max': 5, 'default': 0.5, 'step': 0.1, 'unit': '%'},
    'Lymphocyte': {'label': 'Lymphocyte', 'type': 'number', 'min': 5, 'max': 50, 'default': 30, 'step': 0.1, 'unit': '%'},
    'PT': {'label': 'Prothrombin Time', 'type': 'number', 'min': 8, 'max': 40, 'default': 12, 'step': 0.1, 'unit': 's'},
    'INR': {'label': 'INR', 'type': 'number', 'min': 0.5, 'max': 5, 'default': 1, 'step': 0.1, 'unit': ''},
    'NT-proBNP': {'label': 'NT-proBNP', 'type': 'number', 'min': 0, 'max': 50000, 'default': 300, 'unit': 'pg/mL'},
    'Creatine kinase': {'label': 'Creatine Kinase', 'type': 'number', 'min': 20, 'max': 2000, 'default': 100, 'unit': 'U/L'},
    'Creatinine': {'label': 'Creatinine', 'type': 'number', 'min': 0.2, 'max': 15, 'default': 1, 'step': 0.1, 'unit': 'mg/dL'},
    'Urea nitrogen': {'label': 'Urea Nitrogen', 'type': 'number', 'min': 5, 'max': 150, 'default': 15, 'unit': 'mg/dL'},
    'glucose': {'label': 'Glucose', 'type': 'number', 'min': 40, 'max': 500, 'default': 100, 'unit': 'mg/dL'},
    'Blood potassium': {'label': 'Blood Potassium', 'type': 'number', 'min': 2, 'max': 8, 'default': 4.2, 'step': 0.1, 'unit': 'mEq/L'},
    'Blood sodium': {'label': 'Blood Sodium', 'type': 'number', 'min': 120, 'max': 160, 'default': 140, 'unit': 'mEq/L'},
    'Blood calcium': {'label': 'Blood Calcium', 'type': 'number', 'min': 6, 'max': 14, 'default': 9, 'step': 0.1, 'unit': 'mg/dL'},
    'Chloride': {'label': 'Chloride', 'type': 'number', 'min': 80, 'max': 130, 'default': 100, 'unit': 'mEq/L'},
    'Anion gap': {'label': 'Anion Gap', 'type': 'number', 'min': 3, 'max': 30, 'default': 12, 'unit': 'mEq/L'},
    'Magnesium ion': {'label': 'Magnesium', 'type': 'number', 'min': 1, 'max': 4, 'default': 2, 'step': 0.1, 'unit': 'mg/dL'},
    'PH': {'label': 'Blood pH', 'type': 'number', 'min': 6.8, 'max': 7.8, 'default': 7.4, 'step': 0.01, 'unit': ''},
    'Bicarbonate': {'label': 'Bicarbonate', 'type': 'number', 'min': 10, 'max': 40, 'default': 24, 'unit': 'mEq/L'},
    'Lactic acid': {'label': 'Lactic Acid', 'type': 'number', 'min': 0.3, 'max': 10, 'default': 1, 'step': 0.1, 'unit': 'mmol/L'},
    'PCO2': {'label': 'PCO2', 'type': 'number', 'min': 20, 'max': 80, 'default': 40, 'unit': 'mmHg'},
    'EF': {'label': 'Ejection Fraction', 'type': 'number', 'min': 10, 'max': 80, 'default': 55, 'unit': '%'},
    'gender': {'label': 'Gender', 'type': 'select', 'options': [('1', 'Male'), ('2', 'Female')], 'default': '1'},
    'hypertensive': {'label': 'Hypertensive', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'atrialfibrillation': {'label': 'Atrial Fibrillation', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'CHD with no MI': {'label': 'CHD without MI', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'diabetes': {'label': 'Diabetes', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'deficiencyanemias': {'label': 'Deficiency Anemia', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'depression': {'label': 'Depression', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'Hyperlipemia': {'label': 'Hyperlipemia', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'Renal failure': {'label': 'Renal Failure', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
    'COPD': {'label': 'COPD', 'type': 'select', 'options': [('0', 'No'), ('1', 'Yes')], 'default': '0'},
}

@app.route('/')
def home():
    """Render the home page with prediction form"""
    load_model()
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
def predict():
    """Make prediction based on input features"""
    try:
        load_model()
        
        # Get input values
        input_data = []
        for feat in selected_features:
            value = request.form.get(feat, 0)
            try:
                input_data.append(float(value))
            except ValueError:
                input_data.append(0)
        
        # Preprocess input
        input_array = np.array(input_data).reshape(1, -1)
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'probability_no_disease': round(probability[0] * 100, 2),
            'probability_disease': round(probability[1] * 100, 2),
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

# Initialize model on startup
load_model()

if __name__ == '__main__':
    print("🫀 CardioML - Heart Disease Prediction")
    print("=" * 40)
    print(f"Selected features: {selected_features}")
    print("=" * 40)
    print("Starting web server...")
    app.run(debug=True, port=5000)
