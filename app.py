import os
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Define paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'Datasets')
MODEL_PATH = os.path.join(BASE_DIR, 'flight_delay_model.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')

# Define the path to the frontend build directory
frontend_dist_path = os.path.abspath(os.path.join(BASE_DIR, '../frontend/dist'))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Pydantic model for input validation
class PredictionRequest(BaseModel):
    departure_airport: str
    destination_airport: str
    airline: str
    departure_time: str
    day: int
    month: int

@app.get("/api/hello")
def hello_world():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/airports")
def get_airports():
    if airports_df is not None:
        result = airports_df[['IATA_CODE', 'AIRPORT']].rename(columns={'IATA_CODE': 'code', 'AIRPORT': 'name'}).to_dict(orient='records')
        return result
    return []

@app.get("/api/airlines")
def get_airlines():
    if airlines_df is not None:
        result = airlines_df[['IATA_CODE', 'AIRLINE']].rename(columns={'IATA_CODE': 'code', 'AIRLINE': 'name'}).to_dict(orient='records')
        return result
    return []

@app.post("/api/predict")
def predict_delay(request: PredictionRequest):
    if not model or not encoders:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # 1. Parse Input
        dep_time_str = request.departure_time # "HH:MM"
        
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
                return 0 
        
        airline_enc = safe_encode(encoders['AIRLINE'], request.airline)
        origin_enc = safe_encode(encoders['ORIGIN_AIRPORT'], request.departure_airport)
        dest_enc = safe_encode(encoders['DESTINATION_AIRPORT'], request.destination_airport)
        
        # 3. Create Feature Vector
        # Features order must match training: ['MONTH', 'DAY', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE']
        features = np.array([[request.month, request.day, airline_enc, origin_enc, dest_enc, dep_time]])
        
        # 4. Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map prediction to risk level
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk = risk_levels.get(prediction, 'Unknown')
        
        return {
            'risk': risk,
            'probability': {
                'low': probabilities[0],
                'medium': probabilities[1] if len(probabilities) > 1 else 0,
                'high': probabilities[2] if len(probabilities) > 2 else 0
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Serve static files (Frontend)
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="static")

    # Fallback for SPA routing (e.g., /about, /contact) - catch-all exception
    # Note: StaticFiles(html=True) handles index.html for root, but not deep links if they don't exist as files.
    # FastAPI's StaticFiles doesn't natively support SPA fallback perfectly without a custom middleware or route.
    # However, for simple use cases, mounting at root is often enough if we don't have conflicting API routes.
    # A common pattern for SPAs in FastAPI:
    
    @app.exception_handler(404)
    async def custom_404_handler(request, exc):
        return FileResponse(os.path.join(frontend_dist_path, 'index.html'))

if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
