# fraud_detector_api.py - Flask API for Fraud Detection

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=" * 60)
print("🚀 FRAUD DETECTION API SERVER")
print("=" * 60)

# Initialize Flask
app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

# Load model components
print("\n📂 Loading trained model and components...")

try:
    model = joblib.load('model.pkl')
    print("   ✅ Model loaded successfully")
    
    scaler = joblib.load('scaler.pkl')
    print("   ✅ Scaler loaded successfully")
    
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)
    feature_columns = model_config['feature_columns']
    print(f"   ✅ Configuration loaded ({len(feature_columns)} features)")
    
    print("\n✅ All components loaded successfully!")
    
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find model files!")
    print(f"   Make sure you've run train_model.py first")
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR loading model: {e}")
    exit(1)

# Define foreign locations
FOREIGN_LOCATIONS = ['London', 'Dubai', 'Tokyo', 'Moscow', 'Lagos']

def engineer_features(transaction_data):
    """Convert raw transaction data into ML features"""
    
    dt = datetime.strptime(transaction_data['DateTime'], '%Y-%m-%d %H:%M:%S')
    
    hour = dt.hour
    day_of_week = dt.weekday()
    day = dt.day
    month = dt.month
    
    amount = float(transaction_data['Amount'])
    
    is_foreign = 1 if transaction_data['Location'] in FOREIGN_LOCATIONS else 0
    card_present_encoded = 1 if transaction_data['CardPresent'] == 'Yes' else 0
    transaction_type_encoded = 1 if transaction_data['TransactionType'] == 'Credit' else 0
    
    category_mapping = {
        'Electronics': 0, 'Entertainment': 1, 'Gas Station': 2,
        'Grocery': 3, 'Healthcare': 4, 'Hotel': 5,
        'Online Retail': 6, 'Restaurant': 7, 'Transportation': 8, 'Travel': 9
    }
    merchant_category_encoded = category_mapping.get(transaction_data['MerchantCategory'], 0)
    
    is_round_amount = 1 if amount in [1000, 2000, 5000, 10000] else 0
    is_high_amount = 1 if amount > 1000 else 0
    is_odd_hour = 1 if (hour >= 2 and hour <= 5) else 0
    
    features = [
        amount, hour, day_of_week, day, month,
        is_foreign, card_present_encoded, transaction_type_encoded,
        merchant_category_encoded, is_round_amount, is_high_amount, is_odd_hour
    ]
    
    return features

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Fraud Detection API is running',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /',
            'predict': 'POST /predict',
            'batch_predict': 'POST /batch_predict'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a single transaction is fraudulent"""
    
    try:
        transaction_data = request.get_json()
        
        required_fields = [
            'TransactionID', 'DateTime', 'Amount', 'MerchantCategory',
            'Location', 'CardPresent', 'TransactionType'
        ]
        
        missing_fields = [field for field in required_fields if field not in transaction_data]
        
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        features = engineer_features(transaction_data)
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        fraud_probability = prediction_proba[1]
        risk_score = int(fraud_probability * 100)
        
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        response = {
            'TransactionID': transaction_data['TransactionID'],
            'prediction': int(prediction),
            'is_fraud': bool(prediction == 1),
            'fraud_probability': round(fraud_probability, 4),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': '1.0'
        }
        
        print(f"[{response['timestamp']}] Transaction {transaction_data['TransactionID']}: "
              f"Risk={risk_level} ({risk_score}%) - "
              f"{'⚠️ FRAUD' if prediction == 1 else '✅ NORMAL'}")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple transactions at once"""
    
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({'error': 'No transactions provided'}), 400
        
        results = []
        
        for trans in transactions:
            try:
                features = engineer_features(trans)
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)
                
                prediction = model.predict(features_scaled)[0]
                fraud_probability = model.predict_proba(features_scaled)[0][1]
                risk_score = int(fraud_probability * 100)
                
                if risk_score >= 80:
                    risk_level = "CRITICAL"
                elif risk_score >= 60:
                    risk_level = "HIGH"
                elif risk_score >= 40:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                results.append({
                    'TransactionID': trans['TransactionID'],
                    'prediction': int(prediction),
                    'is_fraud': bool(prediction == 1),
                    'fraud_probability': round(fraud_probability, 4),
                    'risk_score': risk_score,
                    'risk_level': risk_level
                })
                
            except Exception as e:
                results.append({
                    'TransactionID': trans.get('TransactionID', 'UNKNOWN'),
                    'error': str(e)
                })
        
        total = len(results)
        fraud_count = sum(1 for r in results if r.get('is_fraud', False))
        
        response = {
            'total_transactions': total,
            'fraud_detected': fraud_count,
            'fraud_percentage': round((fraud_count / total) * 100, 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        
        print(f"[BATCH] Processed {total} transactions, {fraud_count} fraud detected")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Batch error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌐 Starting Flask API Server...")
    print("=" * 60)
    print("\n📍 Server Details:")
    print(f"   • URL: http://localhost:5000")
    print(f"   • Health Check: http://localhost:5000/")
    print(f"   • Prediction Endpoint: http://localhost:5000/predict")
    print(f"   • Batch Endpoint: http://localhost:5000/batch_predict")
    print("\n📝 Ready to receive requests from UiPath!")
    print("   Press CTRL+C to stop the server\n")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

    