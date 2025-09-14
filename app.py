from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load separate models for each zone
model_zone1 = pickle.load(open("zone1_consumption_model.pkl", "rb"))
model_zone2 = pickle.load(open("zone2_consumption_model.pkl", "rb"))
model_zone3 = pickle.load(open("zone3_consumption_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from form
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        windspeed = float(request.form["windspeed"])
        gdf = float(request.form["gdf"])
        df = float(request.form["df"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        time_in_min = int(request.form["time_in_min"])
        zone = int(request.form["zone"])  # 1, 2, or 3

        # Arrange features (8 in correct order)
        features = np.array([[temperature, humidity, windspeed, gdf, df,
                              day, month, time_in_min]])

        # Select correct model
        if zone == 1:
            prediction = model_zone1.predict(features)[0]
        elif zone == 2:
            prediction = model_zone2.predict(features)[0]
        else:  # zone == 3
            prediction = model_zone3.predict(features)[0]

        return render_template("index.html",
                               prediction_text=f"Predicted Consumption for Zone {zone}: {prediction:.2f} kWh")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
