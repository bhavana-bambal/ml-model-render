from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "ML Model Running on Render"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    data = np.array(data).reshape(1, -1)
    
    result = model.predict(data)
    
    return jsonify({'result': result.tolist()})

# IMPORTANT for Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)