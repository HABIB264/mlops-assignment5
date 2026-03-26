# check_threshold.py
import mlflow
import sys
import os
import glob

def main():
    # Read the run ID from the artifact
    try:
        with open("model_info.txt", "r") as f:
            run_id = f.read().strip()
        print(f"📖 Read Run ID: {run_id}")
    except FileNotFoundError:
        print("❌ Error: model_info.txt not found")
        sys.exit(1)
    
    # Set tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    print("✅ Using local MLflow tracking at ./mlruns")
    
    # Debug: List what's in mlruns directory
    print("\n📁 Checking mlruns directory:")
    if os.path.exists("./mlruns"):
        for root, dirs, files in os.walk("./mlruns"):
            level = root.replace("./mlruns", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
    else:
        print("❌ mlruns directory does not exist!")
        sys.exit(1)
    
    try:
        # Try to get the run
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print("❌ Error: No accuracy metric found")
            print(f"Available metrics: {list(run.data.metrics.keys())}")
            sys.exit(1)
        
        print(f"\n📊 Model accuracy: {accuracy:.4f}")
        
        # Check threshold
        if accuracy >= 0.85:
            print("✅ Model meets accuracy threshold (>= 0.85)")
            print(f"🐳 Building Docker image for Run ID: {run_id}")
            sys.exit(0)
        else:
            print(f"❌ Model accuracy {accuracy:.4f} is below threshold 0.85")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error fetching MLflow run: {e}")
        print(f"Run ID: {run_id}")
        
        # Try to find the run manually
        print("\n🔍 Searching for run in mlruns directory...")
        try:
            import yaml
            for meta_file in glob.glob("mlruns/**/meta.yaml", recursive=True):
                with open(meta_file, 'r') as f:
                    data = yaml.safe_load(f)
                    if data.get('run_id') == run_id:
                        print(f"✅ Found run data in {meta_file}")
                        print(f"   Run info: {data}")
                        break
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()
