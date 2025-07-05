
import joblib
import pandas as pd
import textstat
import argparse
from flask import Flask, request, jsonify

# Load model
model = joblib.load("baseline_retrained_model_20k.pkl")

# Encode suspicion levels
suspicion_map = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Feature extraction function
def extract_features(message):
    text = message.get("Text", "")
    trust_score = float(message.get("Trust Score", 0.8))
    suspicion_score = float(message.get("Suspicion Score", 0.2))
    suspicion_level = message.get("Suspicion Level", "none").lower()
    boost = {"low": 0.05, "medium": 0.10, "high": 0.20}.get(suspicion_level, 0.0)
    adjusted_suspicion = min(suspicion_score + boost, 1.0)
    sus_encoded = suspicion_map.get(suspicion_level, 0)

    return {
        "Trust Score": trust_score,
        "Suspicion Score": suspicion_score,
        "Adjusted Suspicion Score": adjusted_suspicion,
        "Suspicion Encoded": sus_encoded,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_score": textstat.dale_chall_readability_score(text)
    }

# Predict from raw message
def predict_message(message):
    features = extract_features(message)
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return "likely honest" if prediction else "likely dishonest"

# CLI interface
def cli_interface():
    parser = argparse.ArgumentParser(description="Baseline Message Veracity Prediction")
    parser.add_argument("--text", type=str, required=True, help="Message text")
    parser.add_argument("--trust", type=float, default=0.8, help="Trust score")
    parser.add_argument("--suspicion", type=float, default=0.2, help="Suspicion score")
    parser.add_argument("--level", type=str, default="none", help="Suspicion level (none, low, medium, high)")
    args = parser.parse_args()

    message = {
        "Text": args.text,
        "Trust Score": args.trust,
        "Suspicion Score": args.suspicion,
        "Suspicion Level": args.level
    }

    verdict = predict_message(message)
    print(f"Prediction: {verdict}")

# API server
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def api_predict():
    message = request.json
    verdict = predict_message(message)
    return jsonify({"verdict": verdict})

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cli_interface()
    else:
        app.run(host="0.0.0.0", port=8080)
