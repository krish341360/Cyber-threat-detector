import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle as pk
from sklearn.ensemble import RandomForestClassifier


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
    X = df.drop('attack_detected', axis=1)  # Features
    y = df['attack_detected']  # Target
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and necessary components
    model_data = {
        "model": model,
        "encoders": encoders,
        "scaler": scaler,
        "feature_names": X.columns.tolist()
    }
    
    with open('trained_model.pkl', 'wb') as f:
        pk.dump(model_data, f)
    
    return X_train, X_test, y_train, y_test, encoders, scaler

# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_data("/Users/krishsharma/Desktop/ThreatDetector/data/cybersecurity_intrusion_data.csv")