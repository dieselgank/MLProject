import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    # Aktifkan autologging dari MLflow
    mlflow.sklearn.autolog()

    # URL dataset
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    # Load data
    df = pd.read_csv(DATA_URL, sep=';')

    # Preprocessing
    df['quality_category'] = np.where(df['quality'] >= 7, 1, 0)
    X = df.drop(['quality', 'quality_category'], axis=1)
    y = df['quality_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training dengan MLflow Run Context
    with mlflow.start_run() as run:
        model = LogisticRegression(C=0.5, random_state=42) # Tambah parameter untuk autolog
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {accuracy:.4f}")
        print("Autologging selesai. Cek MLflow UI.")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()