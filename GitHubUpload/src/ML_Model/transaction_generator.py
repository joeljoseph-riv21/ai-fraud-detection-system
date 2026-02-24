import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration
NUM_TRANSACTIONS = 1000  # Total transactions to generate
FRAUD_PERCENTAGE = 0.1   # 10% will be fraudulent
START_DATE = datetime(2024, 1, 1)  # Start date for transactions

# Realistic merchant names
merchants = [
    "Amazon", "Walmart", "Target", "Starbucks", "McDonald's",
    "Shell Gas Station", "Best Buy", "Home Depot", "CVS Pharmacy",
    "Whole Foods", "Netflix", "Spotify", "Uber", "Lyft",
    "Delta Airlines", "Hilton Hotels", "Booking.com"
]

# Merchant categories
categories = [
    "Online Retail", "Grocery", "Restaurant", "Gas Station",
    "Electronics", "Healthcare", "Entertainment", "Travel",
    "Hotel", "Transportation"
]

# Locations (cities)
normal_locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
foreign_locations = ["London", "Dubai", "Tokyo", "Moscow", "Lagos"]

# Account numbers (simulated)
account_numbers = [f"ACC{str(i).zfill(6)}" for i in range(1, 101)]

def generate_transaction(transaction_id, is_fraud=False):
    """
    Generate a single transaction (normal or fraudulent)
    
    Args:
        transaction_id: Unique ID for this transaction
        is_fraud: Boolean - should this be fraudulent?
    
    Returns:
        Dictionary with all transaction details
    """

    # Pick random account
    account = random.choice(account_numbers)

    # Generate timestamp (random time in past 30 days)
    days_ago = random.randint(0, 30)
    hours = random.randint(0, 23)
    minutes = random.randint(0, 59)
    transaction_time = START_DATE + timedelta(days=days_ago, hours=hours, minutes=minutes)

    if not is_fraud:
        # NORMAL TRANSACTION PATTERN
        merchant = random.choice(merchants)
        category = random.choice(categories)
        location = random.choice(normal_locations)

        # Normal amounts: $5 to $500, with most being small
        if random.random() < 0.7:  # 70% are small purchases
            amount = round(random.uniform(5, 100), 2)
        else:  # 30% are larger purchases
            amount = round(random.uniform(100, 500), 2)

        # Normal hours: 8 AM to 10 PM
        if transaction_time.hour < 8 or transaction_time.hour > 22:
            # Adjust to normal hours
            transaction_time = transaction_time.replace(hour=random.randint(8, 22))

        device_id = f"DEV{random.randint(1000, 9999)}"
        ip_address = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        card_present = random.choice(["Yes", "No"])
        transaction_type = random.choice(["Debit", "Credit"])

    else:
        # FRAUDULENT TRANSACTION PATTERN
        # Use a single random value so the fraud sub-patterns are mutually exclusive
        r = random.random()

        # Fraud Pattern 1: High amounts (round numbers)
        if r < 0.4:  # 40% of fraud is high amounts
            amount = random.choice([1000, 2000, 5000, 10000])
            merchant = random.choice(merchants)
            category = "Online Retail"
            location = random.choice(normal_locations)

        # Fraud Pattern 2: Foreign location
        elif r < 0.7:  # next 30% (0.4 - 0.7)
            amount = round(random.uniform(200, 2000), 2)
            merchant = random.choice(merchants)
            category = random.choice(categories)
            location = random.choice(foreign_locations)

        # Fraud Pattern 3: Odd hours (2 AM - 5 AM)
        else:  # remaining 30%
            amount = round(random.uniform(100, 1000), 2)
            merchant = random.choice(merchants)
            category = random.choice(categories)
            transaction_time = transaction_time.replace(hour=random.randint(2, 5))
            location = random.choice(normal_locations)

        # Fraudulent transactions usually:
        device_id = f"DEV{random.randint(5000, 9999)}"  # Different device
        ip_address = f"10.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"  # Suspicious IP range
        card_present = "No"  # Usually online/card-not-present
        transaction_type = "Credit"

    # Build the transaction dictionary
    return {
        'TransactionID': f"TXN{str(transaction_id).zfill(8)}",
        'AccountNumber': account,
        'DateTime': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Amount': amount,
        'MerchantName': merchant,
        'MerchantCategory': category,
        'Location': location,
        'DeviceID': device_id,
        'IPAddress': ip_address,
        'CardPresent': card_present,
        'TransactionType': transaction_type,
        'IsFraud': 1 if is_fraud else 0  # Label: 1=Fraud, 0=Normal
    }

# Main generation logic
def generate_dataset():
    """Generate complete dataset with normal and fraudulent transactions"""
    
    transactions = []
    num_fraud = int(NUM_TRANSACTIONS * FRAUD_PERCENTAGE)  # 10% = 100 fraud
    num_normal = NUM_TRANSACTIONS - num_fraud  # 90% = 900 normal
    
    print(f"Generating {NUM_TRANSACTIONS} transactions...")
    print(f"  - Normal: {num_normal}")
    print(f"  - Fraudulent: {num_fraud}")
    
    # Generate normal transactions
    for i in range(num_normal):
        transactions.append(generate_transaction(i + 1, is_fraud=False))
    
    # Generate fraudulent transactions
    for i in range(num_fraud):
        transactions.append(generate_transaction(num_normal + i + 1, is_fraud=True))
    
    # Convert to DataFrame (Excel-like table)
    df = pd.DataFrame(transactions)
    
    # Shuffle rows (mix fraud and normal randomly)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate and save
if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset()
    
    # Save to CSV
    output_file = "../Data/transactions.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Dataset saved to: {output_file}")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud count: {df['IsFraud'].sum()}")
    print(f"Fraud percentage: {(df['IsFraud'].sum() / len(df)) * 100:.1f}%")
    
    # Show first few rows
    print("\nSample transactions:")
    print(df.head())

