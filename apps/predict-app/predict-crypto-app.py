import numpy as np
from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from pydantic import BaseModel

# Load preprocessed data
data = pd.read_csv("../data/preprocessed_crypto_data_ml.csv")

# Encode 'order_type' using one-hot encoding
data = pd.get_dummies(data, columns=['order_type'], drop_first=True)

# Split features and target
X = data.drop(columns=["amount"])  # Features
y = data["amount"]  # Target

# Ensure all features are numeric
if not all([pd.api.types.is_numeric_dtype(col) for col in X.dtypes]):
    raise ValueError("All feature columns must be numeric after encoding.")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
with open("linear_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Initialize FastAPI
app = FastAPI()

# Define the input schema using Pydantic
class PredictionRequest(BaseModel):
    pair: str

# Load the model (ensure this file exists)
try:
    with open("linear_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError("Model file 'linear_model.pkl' not found. Train the model first.")

@app.post("/predict")
def predict(request: PredictionRequest):
    # Define default values for other features
    default_rate = 15300000.0
    default_ingest_year = 2024
    default_ingest_month = 11
    default_ingest_day = 23
    default_ingest_hour = 20

    # Prepare input for prediction
    input_data = pd.DataFrame([{
        "rate": default_rate,
        "ingest_year": default_ingest_year,
        "ingest_month": default_ingest_month,
        "ingest_day": default_ingest_day,
        "ingest_hour": default_ingest_hour,
        **{f"pair_{request.pair}": 1}  # One-hot encoding for the pair
    }])

    # Fill missing columns for one-hot encoding
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # Ensure the column order matches the model's expectations
    input_data = input_data[model.feature_names_in_]

    # Predict
    try:
        prediction = model.predict(input_data)[0]
        return {"predicted_amount": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)