
from flask import Flask, render_template, request
import pickle
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "rf_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Features used during model training
FEATURE_COLUMNS = [
    "Age", "Pclass", "Sex", "SibSp",
    "Parch", "Fare", "Embarked"
]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        pclass = int(request.form["pclass"])
        sex = int(request.form["sex"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        embarked = int(request.form["embarked"])

        # Collect passenger information
        input_data = pd.DataFrame(
            [[age, pclass, sex, sibsp, parch, fare, embarked]],
            columns=FEATURE_COLUMNS
        )

        # Make prediction
        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            result = "Survived"
            message = "The passenger is predicted to have survived."
        else:
            result = "Did Not Survive"
            message = "The passenger is predicted not to have survived."

        # Display result on the same page
        return render_template(
            "index.html",
            result=result,
            message=message
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numbers for all fields."
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Please check your model and input data."
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
