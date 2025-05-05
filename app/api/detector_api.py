from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from joblib import load
import pandas as pd
# from flask_cors import CORS  # For cross-origin requests
import pickle as pk

app = Flask(__name__)
# CORS(app)  # Enable CORS if using a frontend
api = Api(app)
with open('trained_model.pkl', 'rb') as f:
    # Load the trained model
    model_data = pk.load(f)
model = model_data["model"]  # Extract the model
feature_names = model_data["feature_names"] 


FEATURES = [
    'protocol_type', 'encryption_used', 'browser_type',
    'network_packet_size', 'login_attempts', 'session_duration',
    'ip_reputation_score', 'failed_logins', 'unusual_time_access'
]

class ThreatDetector(Resource):
    def post(self):
        data = request.get_json()
        
        # Validate input features
        missing = [feat for feat in feature_names if feat not in data]
        if missing:
            return {"error": f"Missing features: {missing}"}, 400
        
        # Reorder input to match training data
        input_data = [data[feat] for feat in feature_names]
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Predict
        prediction = model.predict(input_df)
        return {"threat_detected": bool(prediction[0])}

api.add_resource(ThreatDetector, '/detect')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Turn off debug in production
