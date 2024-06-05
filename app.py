from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
pipeline = joblib.load('models/pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = pipeline.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
