# Cybersecurity Threat Detector

A machine learning-based system to detect cybersecurity threats from network traffic and system data.

---

## Project Overview

Cyber threats are increasingly sophisticated, making automated detection critical. This project uses a Random Forest classifier to analyze network features and predict whether an event is malicious. The system includes:

- **Data preprocessing**: Cleans and encodes raw network data.
- **Model training**: Trains a Random Forest on labeled intrusion data.
- **REST API**: A Flask-based API to serve real-time threat predictions.
- **Evaluation**: Metrics to assess model accuracy and reliability.

---

## Dataset

The model is trained on a labeled dataset containing features such as:

- `protocol_type` (e.g., TCP, UDP, ICMP)
- `encryption_used` (e.g., AES, DES, None)
- `browser_type` (e.g., Chrome, Firefox, Edge)
- Numerical features like `network_packet_size`, `login_attempts`, `session_duration`, `ip_reputation_score`, `failed_logins`, `unusual_time_access`
- Target label: `attack_detected` (0 = benign, 1 = malicious)

The dataset file `cybersecurity_intrusion_data.csv` is located in the `data/` directory.

---

## Project Structure

Cyber-threat-detector/\n
├── app/
│ ├── api/
│ │ └── detector_api.py # Flask REST API serving the model
│ ├── data_processing.py # Data cleaning and feature engineering
│ └── model.py # Model training and saving
├── data/
│ └── cybersecurity_intrusion_data.csv # Dataset CSV file
├── models/
│ └── trained_model.pkl # Serialized trained model and metadata
├── tests/
│ └── test_model.py # Unit tests for model functionality
├── requirements.txt # Python dependencies
└── README.md # This file


---

## Installation & Setup

1. **Clone the repo:**
git clone https://github.com/krish341360/Cyber-threat-detector.git
cd Cyber-threat-detector


2. **Create and activate a virtual environment:**
python3 -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate.bat # Windows


3. **Install dependencies:**
pip install -r requirements.txt


4. **Prepare your dataset:**
Place your `cybersecurity_intrusion_data.csv` file inside the `data/` folder.

5. **Train the model:**
python app/model.py

This script loads and preprocesses the data, trains the Random Forest model, and saves it as `trained_model.pkl`.

6. **Run the API server:**
python app/api/detector_api.py

The API will be available at `http://localhost:5000/detect`.

---

## Usage

### Making Predictions via API

Send a POST request with JSON payload containing the required features:

{
"protocol_type": "TCP",
"encryption_used": "AES",
"browser_type": "Chrome",
"network_packet_size": 600,
"login_attempts": 3,
"session_duration": 450.5,
"ip_reputation_score": 0.3,
"failed_logins": 2,
"unusual_time_access": 1
}


**Example using curl:**

curl -X POST http://localhost:5000/detect
-H "Content-Type: application/json"
-d '{
"protocol_type": "TCP",
"encryption_used": "AES",
"browser_type": "Chrome",
"network_packet_size": 600,
"login_attempts": 3,
"session_duration": 450.5,
"ip_reputation_score": 0.3,
"failed_logins": 2,
"unusual_time_access": 1
}'


**Expected response:**

{
"threat_detected": true
}


---

## Testing

Run unit tests to verify model loading, prediction, and saving:

pytest tests/test_model.py -v


---

## How It Works

- **Data Processing:**  
  The `data_processing.py` script loads the dataset, drops irrelevant columns like `session_id`, encodes categorical variables (`protocol_type`, `encryption_used`, `browser_type`) using `LabelEncoder`, scales numerical features with `StandardScaler`, and prepares the data for training.

- **Model Training:**  
  The `model.py` script loads and preprocesses data, splits it into training and test sets, trains a Random Forest classifier, and saves the model and feature metadata (`feature_names`) as a pickle file `trained_model.pkl`.

- **API Serving:**  
  The Flask API in `detector_api.py` loads the saved model and feature names, validates incoming JSON requests to ensure all required features are present, arranges them in the correct order, and returns the model’s prediction as a JSON response.

---

## Future Improvements

- Add live network packet capture integration for real-time monitoring.
- Implement unsupervised anomaly detection for zero-day threat detection.
- Build a frontend dashboard to visualize threat alerts and trends.
- Automate incident response workflows based on detected threats.
- Expand dataset and model to multi-class classification for different attack types.

---

## Dependencies

See `requirements.txt` for all required Python packages, including:

- `scikit-learn`
- `pandas`
- `numpy`
- `flask`
- `joblib`

---

## License

This project is licensed under the MIT License.

---

## Contact

Created by [krish341360](https://github.com/krish341360).  
Feel free to open issues or submit pull requests!

---

*Thank you for exploring the Cybersecurity Threat Detector!*
