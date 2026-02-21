# AeroPredictAI Backend

This is the Flask backend for the AeroPredictAI project. It serves a Machine Learning model to predict flight delay risks.

## Setup

1.  **Create virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model:**
    Before running the server, you need to train the model using the provided datasets.
    ```bash
    python train_model.py
    ```
    This will generate `flight_delay_model.pkl` and `label_encoders.pkl`.

4.  **Run the server:**
    ```bash
    python app.py
    ```
    The server will start at `http://127.0.0.1:5000`.

## API Endpoints

### `GET /api/airports`
Returns a list of all airports.
- **Response:** `[{"code": "JFK", "name": "John F. Kennedy International Airport"}, ...]`

### `GET /api/airlines`
Returns a list of all airlines.
- **Response:** `[{"code": "AA", "name": "American Airlines"}, ...]`

### `POST /api/predict`
Predicts the risk of flight delay.
- **Request Body:**
  ```json
  {
    "departure_airport": "JFK",
    "destination_airport": "LAX",
    "airline": "AA",
    "departure_time": "14:30",
    "day": 15,
    "month": 7
  }
  ```
- **Response:**
  ```json
  {
    "risk": "Low",
    "probability": {
      "low": 0.85,
      "medium": 0.10,
      "high": 0.05
    }
  }
  ```

## Connecting to Frontend

There are two ways to connect the frontend:

### 1. Development Mode (Recommended)

In development, run the frontend separately (e.g., using `npm run dev` in the `frontend` folder).
To allow the frontend to make API calls to the backend, configure a proxy in `frontend/vite.config.js`:

```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
```

Then, in your frontend code, you can fetch data from `/api/...`.

### 2. Production Build

Build the frontend:
```bash
cd ../frontend
npm run build
```

The backend is configured to serve the static files from `../frontend/dist`.
Once built, you can access the application at `http://127.0.0.1:5000`.
