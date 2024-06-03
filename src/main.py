import pandas as pd
import joblib
from model_training import evaluate_pipeline, train_pipeline
from data_processing import preprocess_data, load_data
import sys

load_data("data/bike_sharing.csv")
preprocess_data()
train_pipeline()

def validate_pipeline():
    pipeline = joblib.load('models/pipeline.pkl') #loaded_pipeline
    evaluate_pipeline()

if __name__ == "__main__":
    if sys.argv[1] == 'validate_pipeline':
        validate_pipeline()
