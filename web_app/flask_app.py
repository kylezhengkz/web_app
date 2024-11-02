from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
from tensorflow.keras.models import load_model
from model import Model

model = load_model("resources/test_model.h5")
with open("resources/embedding_dictionary.pkl", "rb") as f:
    embedding_dictionary = pickle.load(f)
    
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
