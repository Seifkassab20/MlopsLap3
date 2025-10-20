import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

os.makedirs('models', exist_ok=True)

df = pd.read_csv("data/train.csv")

X = df.drop(columns=['loan_approved', 'name', 'city'])
y = df['loan_approved']

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "models/random_forest.pkl")
print("Training complete")
