from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model on startup
# If the file doesn't exist, it will throw an error, which is expected behavior
# as the user must provide the model file.
try:
    model = joblib.load("solar_power_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model. {e}")
    model = None

# Column names expected by the model
FEATURE_COLUMNS = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'Hour', 'Day', 'Month']

def get_status_message(power):
    """Returns a short status message based on the predicted power output."""
    if power <= 50:
        return "Low output — minimal charging, conserve battery usage."
    elif power <= 500:
        return "Moderate output — normal usage recommended."
    else:
        return "High output — good time for charging batteries and running appliances."

@app.route("/")
def index():
    """Renders the main HTML page."""
    return render_template("index.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    Accepts 6 input values, validates them, and runs predictions.
    Also calculates predictions for all 24 hours of the given day.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please ensure solar_power_model.pkl is present."}), 500

    try:
        # Extract inputs from the JSON request
        data = request.json
        
        # Parse and validate inputs
        amb_temp = float(data.get("ambient_temp", 25.0))
        mod_temp = float(data.get("module_temp", 25.0))
        irradiation = float(data.get("irradiation", 0.0))
        hour = int(data.get("hour", 12))
        day = int(data.get("day", 1))
        month = int(data.get("month", 1))

        # 1. Prediction for the specific input
        input_data = {
            'AMBIENT_TEMPERATURE': [amb_temp],
            'MODULE_TEMPERATURE': [mod_temp],
            'IRRADIATION': [irradiation],
            'Hour': [hour],
            'Day': [day],
            'Month': [month]
        }
        df = pd.DataFrame(input_data)[FEATURE_COLUMNS]
        
        # Run prediction and clip to 0
        predicted_power = model.predict(df)[0]
        predicted_power = max(0.0, float(predicted_power))
        
        # Generate status message
        status_message = get_status_message(predicted_power)

        # 24-hour prediction series (Bonus)
        # Apply a diurnal scaling factor to create a realistic bell curve (peak at noon, zero at night)
        import math
        hours_list = list(range(24))
        
        irr_list = []
        amb_list = []
        mod_list = []
        
        for h in hours_list:
            if 6 <= h <= 18:
                # Bell curve peaking at hour 12
                scale = math.sin((h - 6) * math.pi / 12)
            else:
                scale = 0.0
                
            irr_list.append(irradiation * scale)
            amb_list.append(amb_temp - 5 + (5 * scale)) # Temp drops a bit at night
            mod_list.append(mod_temp - 10 + (10 * scale)) # Mod temp drops more at night

        hourly_data = {
            'AMBIENT_TEMPERATURE': amb_list,
            'MODULE_TEMPERATURE': mod_list,
            'IRRADIATION': irr_list,
            'Hour': hours_list,
            'Day': [day] * 24,
            'Month': [month] * 24
        }
        hourly_df = pd.DataFrame(hourly_data)[FEATURE_COLUMNS]
        
        # Predict for all 24 hours
        hourly_predictions = model.predict(hourly_df)
        hourly_predictions = [max(0.0, float(p)) for p in hourly_predictions]

        return jsonify({
            "success": True,
            "predicted_power": predicted_power,
            "status_message": status_message,
            "hourly_predictions": hourly_predictions,
            "hours": hours_list
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid input format. Please ensure all fields are numbers. Details: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
