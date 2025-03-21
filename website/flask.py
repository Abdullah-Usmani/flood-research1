from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained LSTM model
model = tf.keras.models.load_model('path_to_your_lstm_model.h5')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Extract the relevant weather data
    weather_data = data["weatherData"]
    
    # Assume the LSTM model expects certain variables (reshape accordingly)
    temperature_max = weather_data['temperature_2m_max']
    temperature_mean = weather_data['temperature_2m_mean']
    temperature_min = weather_data['temperature_2m_min']
    daylight_duration = weather_data['daylight_duration']
    precipitation_sum = weather_data['precipitation_sum']
    shortwave_radiation_sum = weather_data['shortwave_radiation_sum']
    
    # Convert the daily weather data into a format suitable for your LSTM model
    input_data = np.array([temperature_max, temperature_mean, temperature_min, daylight_duration, precipitation_sum, shortwave_radiation_sum]).T
    input_data = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))  # Reshaping for LSTM

    # Make prediction
    prediction = model.predict(input_data)

    return jsonify({"streamflow": float(prediction[0][0])})

if __name__ == "__main__":
    app.run(debug=True)
