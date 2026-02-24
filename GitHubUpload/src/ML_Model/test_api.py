# test_api.py - API Testing Script

import requests
import json

API_URL = "http://localhost:5000/predict"

# Test case 1: Normal transaction
normal_transaction = {
    "TransactionID": "TEST_NORMAL_001",
    "AccountNumber": "ACC000042",
    "DateTime": "2024-01-15 14:30:00",
    "Amount": 25.50,
    "MerchantName": "Starbucks",
    "MerchantCategory": "Restaurant",
    "Location": "New York",
    "DeviceID": "DEV1234",
    "IPAddress": "192.168.1.50",
    "CardPresent": "Yes",
    "TransactionType": "Debit"
}

# Test case 2: Fraudulent transaction
fraud_transaction = {
    "TransactionID": "TEST_FRAUD_001",
    "AccountNumber": "ACC000042",
    "DateTime": "2024-01-15 03:15:00",
    "Amount": 10000,
    "MerchantName": "Best Buy",
    "MerchantCategory": "Electronics",
    "Location": "Tokyo",
    "DeviceID": "DEV9876",
    "IPAddress": "10.50.100.200",
    "CardPresent": "No",
    "TransactionType": "Credit"
}

print("=" * 60)
print("🧪 TESTING FRAUD DETECTION API")
print("=" * 60)

# Test normal transaction
print("\n📊 Test 1: Normal Transaction")
print("-" * 60)
response = requests.post(API_URL, json=normal_transaction)
result = response.json()
print(f"Transaction ID: {result['TransactionID']}")
print(f"Prediction: {'FRAUD' if result['is_fraud'] else 'NORMAL'}")
print(f"Risk Score: {result['risk_score']}%")
print(f"Risk Level: {result['risk_level']}")

# Test fraud transaction
print("\n📊 Test 2: Fraudulent Transaction")
print("-" * 60)
response = requests.post(API_URL, json=fraud_transaction)
result = response.json()
print(f"Transaction ID: {result['TransactionID']}")
print(f"Prediction: {'FRAUD' if result['is_fraud'] else 'NORMAL'}")
print(f"Risk Score: {result['risk_score']}%")
print(f"Risk Level: {result['risk_level']}")

print("\n" + "=" * 60)
print("✅ API TESTS COMPLETE!")
print("=" * 60)
