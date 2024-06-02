import joblib
import sys

def save_pipeline():
    with open('models/pipeline.pkl', 'rb') as f:
        pipeline = joblib.load(f)
    with open('models/saved_pipeline.pkl', 'wb') as f:
        joblib.dump(pipeline, f)

def load_pipeline():
    with open('models/saved_pipeline.pkl', 'rb') as f:
        pipeline = joblib.load(f)
    with open('models/loaded_pipeline.pkl', 'wb') as f:
        joblib.dump(pipeline, f)

if __name__ == "__main__":
    if sys.argv[1] == 'save_pipeline':
        save_pipeline()
    elif sys.argv[1] == 'load_pipeline':
        load_pipeline()
