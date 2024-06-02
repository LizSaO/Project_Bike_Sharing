import joblib

def save_pipeline(pipeline, file_path):
    joblib.dump(pipeline, file_path)

def load_pipeline(file_path):
    return joblib.load(file_path)
