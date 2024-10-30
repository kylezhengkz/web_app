# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def model(data):
    return {"result": f"{data['input']}"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
