import joblib
import numpy as np

with open("./model/best_loan_model.pkl", "rb") as file:
    scaler, model = joblib.load(file)

def predict(data: dict):
    # convert input to array
    input_data = np.array([list(data.values())])

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
