from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Ensure same columns
    missing_cols = set(model_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[model_columns]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]

    return jsonify({"prediction": int(prediction), "probability": float(proba)})

if __name__ == "__main__":
    app.run(debug=True)
