from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from joblib import load
import pandas as pd
import pickle as pk
import numpy as np

app = Flask(__name__)
api = Api(app)

# Load the model and encoders
with open('trained_model.pkl', 'rb') as f:
    model_data = pk.load(f)
model = model_data["model"]
feature_names = model_data["feature_names"]
encoders = model_data["encoders"]
scaler = model_data["scaler"]

FEATURES = [
    'protocol_type', 'encryption_used', 'browser_type',
    'network_packet_size', 'login_attempts', 'session_duration',
    'ip_reputation_score', 'failed_logins', 'unusual_time_access'
]

class ThreatDetector(Resource):
    def post(self):
        try:
            data = request.get_json()
            
            # Validate input features
            missing = [feat for feat in feature_names if feat not in data]
            if missing:
                return {"error": f"Missing features: {missing}"}, 400
            
            # Create a DataFrame with the input data
            input_df = pd.DataFrame([data])
            
            # Encode categorical features
            categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
            for col in categorical_cols:
                if col in encoders:
                    input_df[col] = encoders[col].transform([data[col]])[0]
            
            # Scale numerical features
            numerical_cols = [
                'network_packet_size', 'login_attempts', 'session_duration',
                'ip_reputation_score', 'failed_logins', 'unusual_time_access'
            ]
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Predict
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]  # Get probability of threat
            
            return {
                "threat_detected": bool(prediction[0]),
                "threat_probability": float(probability)
            }
            
        except Exception as e:
            return {"error": str(e)}, 500

api.add_resource(ThreatDetector, '/detect')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)  
