from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("shopping_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = float(request.form["age"])
        income = float(request.form["income"])
        pages = float(request.form["pages"])
        time_spent = float(request.form["time_spent"])

        features = np.array([[age, income, pages, time_spent]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "🛍 Customer Will Purchase"
            status = "success"
        else:
            result = "❌ Customer Will Not Purchase"
            status = "danger"

        return render_template(
            "index.html",
            prediction=result,
            status=status
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            status="danger"
        )

if __name__ == "__main__":
    app.run(debug=True)