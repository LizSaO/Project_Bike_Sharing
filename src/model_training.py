import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import sys
import json
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def create_pipeline(preprocessor):
    lm = LinearRegression()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', RFE(estimator=lm, n_features_to_select=25)),
        ('regression', lm)
    ])
    return pipeline

def train_pipeline():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
   # preprocessor = create_preprocessor()
   # pipeline = create_pipeline(preprocessor) 
    pipeline.fit(X_train, y_train)
    with open('model/pipeline.pkl', 'wb') as f:
        joblib.dump(pipeline, f)
    return pipeline

def evaluate_pipeline():
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    with open('model/pipeline.pkl', 'rb') as f:
        pipeline = joblib.load(f)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'mse': mse, 'r2': r2}
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    if sys.argv[1] == 'train_pipeline':
        train_pipeline()
    elif sys.argv[1] == 'evaluate_pipeline':
        evaluate_pipeline()
