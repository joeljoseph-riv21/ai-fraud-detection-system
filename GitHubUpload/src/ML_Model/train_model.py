# train_model.py - Fraud Detection Model Training Script

# Data handling
import pandas as pd
import numpy as np
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Model persistence
import joblib

# Warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 FRAUD DETECTION MODEL TRAINING")
print("=" * 60)

# Step 1: Load the dataset
print("\n📂 Step 1: Loading transaction data...")
df = pd.read_csv('../Data/transactions.csv')

print(f"✅ Loaded {len(df)} transactions")
print(f"   - Fraud cases: {df['IsFraud'].sum()}")
print(f"   - Normal cases: {len(df) - df['IsFraud'].sum()}")
print(f"   - Columns: {len(df.columns)}")


print("\n🔧 Step 2: Feature Engineering...")

# Create a copy to avoid modifying original
df_processed = df.copy()

# Convert DateTime to datetime object
df_processed['DateTime'] = pd.to_datetime(df_processed['DateTime'])

# Extract time-based features
df_processed['Hour'] = df_processed['DateTime'].dt.hour
df_processed['DayOfWeek'] = df_processed['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
df_processed['Day'] = df_processed['DateTime'].dt.day
df_processed['Month'] = df_processed['DateTime'].dt.month

print("   ✓ Extracted time features: Hour, DayOfWeek, Day, Month")

# Check if location is foreign
foreign_locations = ['London', 'Dubai', 'Tokyo', 'Moscow', 'Lagos']
df_processed['IsForeignLocation'] = df_processed['Location'].apply(
    lambda x: 1 if x in foreign_locations else 0
)

print("   ✓ Created binary feature: IsForeignLocation")

# Encode CardPresent (Yes/No → 1/0)
df_processed['CardPresent_Encoded'] = df_processed['CardPresent'].apply(
    lambda x: 1 if x == 'Yes' else 0
)

print("   ✓ Encoded CardPresent: Yes=1, No=0") 

# Encode TransactionType (Credit/Debit)
df_processed['TransactionType_Encoded'] = df_processed['TransactionType'].apply(
    lambda x: 1 if x == 'Credit' else 0
)

print("   ✓ Encoded TransactionType: Credit=1, Debit=0") 

# Encode MerchantCategory using LabelEncoder
le_category = LabelEncoder()
df_processed['MerchantCategory_Encoded'] = le_category.fit_transform(df_processed['MerchantCategory'])

print(f"   ✓ Encoded MerchantCategory into {len(le_category.classes_)} numeric values")
print(f"     Categories: {list(le_category.classes_)}")

# Create amount-based features
df_processed['IsRoundAmount'] = df_processed['Amount'].apply(
    lambda x: 1 if x in [1000, 2000, 5000, 10000] else 0
)

df_processed['IsHighAmount'] = df_processed['Amount'].apply(
    lambda x: 1 if x > 1000 else 0
)

print("   ✓ Created amount features: IsRoundAmount, IsHighAmount")

# Create hour-based risk feature
df_processed['IsOddHour'] = df_processed['Hour'].apply(
    lambda x: 1 if (x >= 2 and x <= 5) else 0
)

print("   ✓ Created time feature: IsOddHour (2 AM - 5 AM)")

print(f"\n📊 Total features created: {len(df_processed.columns)}")


print("\n🎯 Step 3: Selecting features for model training...")

# Define which columns to use for ML
feature_columns = [
    'Amount',                      # Transaction amount
    'Hour',                        # Hour of day
    'DayOfWeek',                   # Day of week
    'Day',                         # Day of month
    'Month',                       # Month
    'IsForeignLocation',           # Foreign location flag
    'CardPresent_Encoded',         # Card present flag
    'TransactionType_Encoded',     # Transaction type
    'MerchantCategory_Encoded',    # Merchant category
    'IsRoundAmount',               # Round amount flag
    'IsHighAmount',                # High amount flag
    'IsOddHour'                    # Odd hour flag
]

# Create feature matrix (X) and target variable (y)
X = df_processed[feature_columns]
y = df_processed['IsFraud']

print(f"✅ Selected {len(feature_columns)} features:")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i}. {col}")

print(f"\n📊 Data shape:")
print(f"   Features (X): {X.shape}")
print(f"   Labels (y): {y.shape}")


print("\n✂️ Step 4: Splitting data into train and test sets...")

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Fixed seed for reproducibility
    stratify=y          # Maintain fraud ratio in both sets
)

print(f"✅ Data split complete:")
print(f"   Training set: {len(X_train)} transactions")
print(f"   Testing set: {len(X_test)} transactions")
print(f"   Training fraud cases: {y_train.sum()}")
print(f"   Testing fraud cases: {y_test.sum()}")


print("\n⚖️ Step 5: Normalizing features...")

# Initialize scaler
scaler = StandardScaler()

# Fit scaler on training data only, then transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features normalized using StandardScaler")
print(f"   Mean: ~0, Standard Deviation: ~1") 


print("\n🌲 Step 6: Training Random Forest Classifier...")

# Initialize the model
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees in forest
    max_depth=10,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples in leaf node
    random_state=42,         # For reproducibility
    n_jobs=-1,               # Use all CPU cores
    class_weight='balanced'  # Handle imbalanced data (10% fraud)
)

# Train the model
print("   Training in progress...")
model.fit(X_train_scaled, y_train)

print("✅ Model training complete!")
print(f"   Model: Random Forest with {model.n_estimators} trees")


print("\n🔮 Step 7: Making predictions on test set...")

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Get probability scores
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of fraud

print(f"✅ Predictions generated for {len(y_pred)} test transactions")


print("\n📊 Step 8: Evaluating model performance...")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 60)
print("📈 MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.2%}  - Overall correctness")
print(f"Precision: {precision:.2%}  - When we predict fraud, how often correct?")
print(f"Recall:    {recall:.2%}  - Of all actual frauds, how many did we catch?")
print(f"F1-Score:  {f1:.2%}  - Balance of precision and recall")
print(f"ROC-AUC:   {roc_auc:.2%}  - Overall model quality")
print("=" * 60)

print("\n🔍 Step 9: Analyzing feature importance...")

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top Features for Fraud Detection:")
print("=" * 50)
for idx, row in feature_importance.iterrows():
    bar_length = int(row['Importance'] * 50)
    bar = '█' * bar_length
    print(f"{row['Feature']:25s} {bar} {row['Importance']:.3f}")
print("=" * 50)


print("\n💾 Step 10: Saving trained model and scaler...")

# Save the trained model
model_path = 'model.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# Save the scaler (needed for predictions later)
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Save feature columns (needed to know feature order)
feature_config = {
    'feature_columns': feature_columns,
    'label_encoder_classes': le_category.classes_.tolist()
}

import json
config_path = 'model_config.json'
with open(config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"✅ Configuration saved to: {config_path}")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📁 Files created:")
print(f"   1. {model_path} - Trained Random Forest model")
print(f"   2. {scaler_path} - Feature scaler")
print(f"   3. {config_path} - Model configuration")
print("\n🚀 Ready for deployment in API!")