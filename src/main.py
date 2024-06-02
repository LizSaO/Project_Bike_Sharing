from data_processing import load_data, preprocess_data, create_preprocessor
from model_training import create_pipeline, train_pipeline, evaluate_model
from model_persistence import save_pipeline, load_pipeline

# Ruta del archivo de datos
file_path = 'path/to/your/bike_sharing.csv'
pipeline_file_path = 'models/bike_sharing_pipeline.pkl'

# Cargar y preprocesar los datos
data = load_data(file_path)
X_train, X_test, y_train, y_test = preprocess_data(data)
preprocessor = create_preprocessor()

# Crear y entrenar el pipeline
pipeline = create_pipeline(preprocessor)
pipeline = train_pipeline(pipeline, X_train, y_train)

# Guardar el pipeline
save_pipeline(pipeline, pipeline_file_path)

# Evaluar el modelo
mse, r2 = evaluate_model(pipeline, X_test, y_test)
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Cargar y validar el pipeline guardado
loaded_pipeline = load_pipeline(pipeline_file_path)
mse_loaded, r2_loaded = evaluate_model(loaded_pipeline, X_test, y_test)
print(f'Loaded MSE: {mse_loaded}')
print(f'Loaded R²: {r2_loaded}')

