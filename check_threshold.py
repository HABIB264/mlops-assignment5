# check_threshold.py
import mlflow
import sys
import os

def main():
    # Read the run ID from the artifact
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("Error: model_info.txt not found")
        sys.exit(1)
    
    print(f"Checking model with Run ID: {run_id}")
    
    # Get the MLflow run
    try:
        # Set tracking URI (this should come from environment variable)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        # Get the run
        run = mlflow.get_run(run_id)
        
        # Extract accuracy metric
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print("Error: No accuracy metric found in the run")
            sys.exit(1)
        
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Check threshold (0.85)
        if accuracy >= 0.85:
            print("✅ Model meets accuracy threshold (>= 0.85)")
            print(f"Building Docker image for Run ID: {run_id}")
            sys.exit(0)  # Success
        else:
            print(f"❌ Model accuracy {accuracy:.4f} is below threshold 0.85")
            sys.exit(1)  # Fail the pipeline
            
    except Exception as e:
        print(f"Error fetching MLflow run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
