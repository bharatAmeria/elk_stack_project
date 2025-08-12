from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import logging
import json
from pythonjsonlogger import jsonlogger
import traceback

# ------------------------
# Configure JSON Logging
# ------------------------
logger = logging.getLogger("elk_logger")
logger.setLevel(logging.INFO)

logHandler = logging.StreamHandler()

# JSON format: timestamp, level, message, and extra fields
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# ------------------------
# Flask App
# ------------------------
app = Flask(__name__)

# Load preprocessed data and model
try:
    data = pd.read_csv("app/processed_data.csv")
    model = load('app/model.pkl')
    logger.info({"event": "MODEL_LOAD_SUCCESS", "message": "Model and data loaded successfully"})
except Exception as e:
    logger.error({
        "event": "MODEL_LOAD_ERROR",
        "error": str(e),
        "trace": traceback.format_exc()
    })
    raise e

@app.route('/')
def index():
    try:
        locations = sorted(data["location"].unique())
        logger.info({"event": "INDEX_PAGE_ACCESS", "locations_count": len(locations)})
        return render_template('index.html', locations=locations)
    except Exception as e:
        logger.error({
            "event": "INDEX_PAGE_ERROR",
            "error": str(e),
            "trace": traceback.format_exc()
        })
        return f"Error loading index page: {e}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        location = request.form.get('location')
        bhk = int(request.form.get('bhk'))
        bath = int(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))

        # Structured logging for request
        logger.info({
            "event": "PREDICTION_REQUEST",
            "location": location,
            "bhk": bhk,
            "bath": bath,
            "sqft": sqft
        })

        # Create input dataframe
        input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                                columns=["location", "total_sqft", "bath", "bhk"])

        # Predict
        prediction = model.predict(input_df)[0] * 1e5

        logger.info({
            "event": "PREDICTION_SUCCESS",
            "predicted_price": round(float(prediction), 2)
        })

        return str(round(float(prediction), 2))

    except Exception as e:
        logger.error({
            "event": "PREDICTION_ERROR",
            "error": str(e),
            "trace": traceback.format_exc()
        })
        return f"Error in prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
