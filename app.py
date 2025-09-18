from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model, scaler, and label encoder
model = joblib.load('gastric_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return "🚀 Gastric AI Monitoring API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON input (features = [MQ4, MQ135, TEMPERATURE, PULSE])
        data = request.get_json(force=True)
        features = data.get('features')

        if not isinstance(features, list) or len(features) != 4:
            return jsonify({'error': 'Invalid input format, expected 4 feature values'}), 400

        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)

        # Apply scaling
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        label = encoder.inverse_transform([prediction])[0]

        # Return prediction result as JSON
        return jsonify({
            'prediction': label,      # Example: "critical" or "normal"
            'raw_class': int(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)