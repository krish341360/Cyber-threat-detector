import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def preprocess_data(filepath):
    df = pd.read_csv("/Users/krishsharma/Desktop/ThreatDetector/data/cybersecurity_intrusion_data.csv")
    df.drop(['session_id'], axis=1, inplace=True)  # Remove session ID
    df.dropna(inplace=True)
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Save encoders for API use
    
    # Scale numerical features
    numerical_cols = [
        'network_packet_size', 'login_attempts', 'session_duration',
        'ip_reputation_score', 'failed_logins', 'unusual_time_access'
    ]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split the data into features and target
    
    return df, encoders, scaler

preprocess_data("/Users/krishsharma/Desktop/ThreatDetector/data/cybersecurity_intrusion_data.csv")