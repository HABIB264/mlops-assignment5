# check_threshold.py
import mlflow
import sys
import os

def main():
    # Read the run ID from the artifact
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
        print(f"📖 Read Run ID: {run_id}")
    except FileNotFoundError:
        print("❌ Error: model_info.txt not found")
        sys.exit(1)
    
    # IMPORTANT: Use the same local tracking path
    mlflow.set_tracking_uri("file:./mlruns")
    print("✅ Using local MLflow tracking at ./mlruns")
    
    try:
        # Get the run
        run = mlflow.get_run(run_id)
        
        # Extract accuracy metric
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print("❌ Error: No accuracy metric found in the run")
            # Try to list all metrics for debugging
            print(f"Available metrics: {list(run.data.metrics.keys())}")
            sys.exit(1)
        
        print(f"📊 Model accuracy: {accuracy:.4f}")
        
        # Check threshold (0.85)
        if accuracy >= 0.85:
            print("✅ Model meets accuracy threshold (>= 0.85)")
            print(f"🐳 Building Docker image for Run ID: {run_id}")
            sys.exit(0)  # Success
        else:
            print(f"❌ Model accuracy {accuracy:.4f} is below threshold 0.85")
            sys.exit(1)  # Fail the pipeline
            
    except Exception as e:
        print(f"❌ Error fetching MLflow run: {e}")
        print(f"Run ID: {run_id}")
        print("Make sure the MLflow run exists in ./mlruns directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
