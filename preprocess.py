import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = ['data/loan_approval.csv']

def find_data():
    for f in data:
        if os.path.isfile(f):
            return f
    raise FileNotFoundError("Data file not found.")

def clean_data(df):
    le = LabelEncoder()
    df['loan_approved'] = le.fit_transform(df['loan_approved'])
    return df
    
def main():
    os.makedirs("data", exist_ok=True)
    data = find_data()
    df = pd.read_csv(data)
    df = clean_data(df)
    train, test = train_test_split(df, test_size=0.3, random_state =42)
    train.to_csv("data/train.csv", index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

