import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

os.makedirs('models', exist_ok=True)

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['loan_approved', 'name', 'city'])
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_estimators_list = [50, 100]
max_depth_list = [3, 5]

with mlflow.start_run(run_name="RandomForest_Tuning") as parent_run:
    best_acc = 0
    best_params = None

    for n in n_estimators_list:
        for d in max_depth_list:
            with mlflow.start_run(run_name=f"n={n}_depth={d}", nested=True):
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

                if acc > best_acc:
                    best_acc = acc
                    best_params = (n, d)
                    mlflow.sklearn.log_model(model, "best_model")
                    model_path = "models/best_model.pkl"
                    mlflow.log_artifact(model_path)
                    mlflow.sklearn.save_model(model, model_path)

    print(f"Best model accuracy: {best_acc:.4f} with n_estimators={best_params[0]}, max_depth={best_params[1]}")
