import pandas as pd
import joblib
from model_training import evaluate_model
import sys

def validate_pipeline():
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    with open('models/loaded_pipeline.pkl', 'rb') as f:
        pipeline = joblib.load(f)
    mse, r2 = evaluate_model(pipeline, X_test, y_test)
    with open('validation/validation_report.txt', 'w') as f:
        f.write(f'Loaded MSE: {mse}\n')
        f.write(f'Loaded R²: {r2}\n')

if __name__ == "__main__":
    if sys.argv[1] == 'validate_pipeline':
        validate_pipeline()
