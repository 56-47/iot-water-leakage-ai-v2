from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained Kaggle model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "water_leakage_model.pkl"
)

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "system": "AI Water Leakage Detection",
        "model": "Random Forest",
        "device": "ESP8266 NodeMCU",
        "endpoint": "/predict",
        "method": "POST",
        "status": "SERVER RUNNING",
        "expected_json": {
            "flow1": 10.5,
            "flow2": 8.2
        }
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "error": "Send a valid JSON request"
            }), 400

        flow1 = float(data["flow1"])
        flow2 = float(data["flow2"])

        if not (
            flow1 >= 0 and flow2 >= 0
            and flow1 < float("inf")
            and flow2 < float("inf")
        ):
            return jsonify({
                "error": "Flow readings must be finite and non-negative"
            }), 400

        difference = abs(flow1 - flow2)

        # Features must match the Kaggle training order
        features = pd.DataFrame([{
            "flow1": flow1,
            "flow2": flow2,
            "difference": difference
        }])

        prediction = str(model.predict(features)[0])

        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)

        return jsonify({
            "flow1": flow1,
            "flow2": flow2,
            "difference": difference,
            "prediction": prediction,
            "status": (
                "POSSIBLE LEAK"
                if prediction == "LEAK"
                else "NORMAL"
            ),
            "confidence": round(confidence, 2)
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide numeric flow1 and flow2 values"
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
