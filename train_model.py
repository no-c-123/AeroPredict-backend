import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Define paths
DATA_DIR = 'Datasets'
FLIGHTS_FILE = os.path.join(DATA_DIR, 'flights.csv')
MODEL_PATH = 'flight_delay_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

def load_data(filepath, sample_size=100000):
    print(f"Loading data from {filepath}...")
    # Load a sample to speed up training for demonstration
    # In production, you might want to use more data or use Dask/Spark
    df = pd.read_csv(filepath, nrows=sample_size) 
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Filter out cancelled and diverted flights
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
    
    # Drop rows with missing target
    df = df.dropna(subset=['ARRIVAL_DELAY'])
    
    # Select features
    features = ['MONTH', 'DAY', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE']
    target = 'ARRIVAL_DELAY'
    
    # Keep only relevant columns
    df = df[features + [target]].copy()
    
    # Create target classes for "Delay Risk"
    # 0: Low Risk (Delay <= 15 mins)
    # 1: Medium Risk (15 < Delay <= 45 mins)
    # 2: High Risk (Delay > 45 mins)
    def classify_delay(delay):
        if delay <= 15:
            return 0
        elif delay <= 45:
            return 1
        else:
            return 2
            
    df['DELAY_RISK'] = df['ARRIVAL_DELAY'].apply(classify_delay)
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Convert to string to handle mixed types if any
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, df['DELAY_RISK'], label_encoders

def train_model():
    # 1. Load Data
    if not os.path.exists(FLIGHTS_FILE):
        print(f"Error: {FLIGHTS_FILE} not found.")
        return

    # Use a larger sample for better accuracy, but keep it manageable
    df = load_data(FLIGHTS_FILE, sample_size=500000)
    
    # 2. Preprocess
    df_processed, y, encoders = preprocess_data(df)
    X = df_processed.drop(columns=['ARRIVAL_DELAY', 'DELAY_RISK'])
    
    print(f"Training with {len(X)} samples...")
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    # RandomForest is good for this, but can be large. Limiting depth to control size.
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    accuracy = clf.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # 6. Save Model and Encoders
    print("Saving model and encoders...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print("Done.")

if __name__ == "__main__":
    train_model()
