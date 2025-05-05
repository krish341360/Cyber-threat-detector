# from app.data_processing import preprocess_data
# from app.model import train_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['session_id'], axis=1)
    X = df.drop('attack_detected', axis=1)
    y = df['attack_detected']

    categorical_cols = ['protocol_type', 'encryption_used', 'browser_type']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    model_data = {
        "model": model,
        "feature_names": X_train.columns.tolist()  # Critical for API
    }
    print("Model trained successfully.")
    return model_data


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data('/Users/krishsharma/Desktop/ThreatDetector/data/cybersecurity_intrusion_data.csv')
    model = train_model(X_train, y_train)
    # print(X_test)
    # Save the model
    import pickle as pk   
    with open('trained_model.pkl', 'wb') as f:
        pk.dump(model, f)
    print("Model saved successfully.")

    # with open('trained_model.pkl', 'rb') as f:
    #     model = pk.load(f)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    # Load the trained model


    # # Predict on the test set
    # y_pred = model.predict(X_test)

    # # Evaluate using common metrics
    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("Precision:", precision_score(y_test, y_pred))
    # print("Recall:", recall_score(y_test, y_pred))
    # print("F1 Score:", f1_score(y_test, y_pred))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
