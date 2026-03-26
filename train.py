# train.py
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    # Simulate loading data (in real scenario, you'd use dvc pull first)
    np.random.seed(42)
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)  # Random will be ~0.5 accuracy
    # Make it predictable for high accuracy
    y = (X[:, 0] > 0.5).astype(int)  # This will have better correlation
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log to MLflow
    with mlflow.start_run() as run:
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Print run ID for debugging
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save run ID to file
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    main()
