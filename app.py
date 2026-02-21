import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

# Define paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
MODEL_PATH = os.path.join(BASE_DIR, 'flight_delay_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# Define the path to the frontend build directory
frontend_dist_path = os.path.abspath(os.path.join(BASE_DIR, '../frontend/dist'))

app = Flask(__name__, static_folder=frontend_dist_path, static_url_path='')
CORS(app) # Enable CORS for all routes

# Load Data and Model
airports_df = None
airlines_df = None
model = None
encoders = None

def load_resources():
    global airports_df, airlines_df, model, encoders
    try:
        print("Loading datasets...")
        airports_df = pd.read_csv(os.path.join(DATA_DIR, 'Airports Data.csv'))
        airlines_df = pd.read_csv(os.path.join(DATA_DIR, 'Airlines Data.csv'))
        
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
            print("Loading model and encoders...")
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
        else:
            print("Model or encoders not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error loading resources: {e}")

load_resources()

@app.route('/api/hello', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello from Flask Backend!'})

@app.route('/api/airports', methods=['GET'])
def get_airports():
    if airports_df is not None:
        # Return a list of dicts: {code: 'JFK', name: 'John F. Kennedy...'}
        result = airports_df[['IATA_CODE', 'AIRPORT']].rename(columns={'IATA_CODE': 'code', 'AIRPORT': 'name'}).to_dict(orient='records')
        return jsonify(result)
    return jsonify([])

@app.route('/api/airlines', methods=['GET'])
def get_airlines():
    if airlines_df is not None:
        result = airlines_df[['IATA_CODE', 'AIRLINE']].rename(columns={'IATA_CODE': 'code', 'AIRLINE': 'name'}).to_dict(orient='records')
        return jsonify(result)
    return jsonify([])

@app.route('/api/predict', methods=['POST'])
def predict_delay():
    if not model or not encoders:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    # Expected keys: departure_airport, destination_airport, airline, departure_time, day, month
    
    try:
        # 1. Parse Input
        dep_airport = data.get('departure_airport')
        dest_airport = data.get('destination_airport')
        airline = data.get('airline')
        dep_time_str = data.get('departure_time') # "HH:MM"
        day = int(data.get('day'))
        month = int(data.get('month'))
        
        # Convert time "HH:MM" to integer HHMM
        if dep_time_str:
            dep_time = int(dep_time_str.replace(':', ''))
        else:
            dep_time = 1200 # Default
            
        # 2. Encode Categorical Variables
        # Helper to safely encode
        def safe_encode(encoder, value):
            try:
                return encoder.transform([value])[0]
            except ValueError:
                # If unseen label, return a default (e.g., 0) or handle appropriately
                # Ideally, we should have an 'unknown' class, but for now we'll use the most frequent (mode) or 0
                return 0 
        
        airline_enc = safe_encode(encoders['AIRLINE'], airline)
        origin_enc = safe_encode(encoders['ORIGIN_AIRPORT'], dep_airport)
        dest_enc = safe_encode(encoders['DESTINATION_AIRPORT'], dest_airport)
        
        # 3. Create Feature Vector
        # Features order must match training: ['MONTH', 'DAY', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE']
        features = np.array([[month, day, airline_enc, origin_enc, dest_enc, dep_time]])
        
        # 4. Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map prediction to risk level
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk = risk_levels.get(prediction, 'Unknown')
        
        return jsonify({
            'risk': risk,
            'probability': {
                'low': probabilities[0],
                'medium': probabilities[1] if len(probabilities) > 1 else 0,
                'high': probabilities[2] if len(probabilities) > 2 else 0
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # Serve index.html for any other route (SPA client-side routing)
        if os.path.exists(os.path.join(app.static_folder, 'index.html')):
             return send_from_directory(app.static_folder, 'index.html')
        else:
            return "Frontend build not found. Please run 'npm run build' in the frontend directory.", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)