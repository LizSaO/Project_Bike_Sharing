import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.to_csv('data/raw_data.csv', index=False)
    return data

def preprocess_data():
    data = pd.read_csv('data/raw_data.csv')
    X = data.drop(columns=['instant', 'dteday', 'cnt','casual', 'registered'])
    y = data['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    return X_train, X_test, y_train, y_test

def create_preprocessor():
    categorical_features = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')
    return preprocessor

if __name__ == "__main__":
    if sys.argv[1] == 'load_data':
        load_data(sys.argv[2])
    elif sys.argv[1] == 'preprocess_data':
        preprocess_data()
