from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def model(data):
    return {"result": f"{data["input"]}"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = model(data)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
