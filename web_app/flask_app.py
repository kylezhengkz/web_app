from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model import Model

myModel = Model()

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = myModel.predict(data["input"])
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
