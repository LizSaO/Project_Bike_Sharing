from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

def create_pipeline(preprocessor):
    lm = LinearRegression()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', RFE(estimator=lm, n_features_to_select=25)),
        ('regression', lm)
    ])
    return pipeline

def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    from sklearn.metrics import mean_squared_error, r2_score
    
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
