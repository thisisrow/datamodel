from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
solubility_model = joblib.load('solubility_model.pkl')
airnoise_model = joblib.load('air-noise.pkl')


# --- Original Solubility Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Solubility prediction endpoint
    Example input:
    {
        "MolLogP": 1.2,
        "MolWt": 250.3,
        "NumRotatableBonds": 3,
        "AromaticProportion": 0.25
    }
    """
    try:
        data = request.get_json()
        input_data = pd.DataFrame([[
            float(data['MolLogP']),
            float(data['MolWt']),
            int(data['NumRotatableBonds']),
            float(data['AromaticProportion'])
        ]], columns=['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion'])

        prediction = solubility_model.predict(input_data)[0]
        return jsonify({'logS': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- Updated Air & Noise Endpoint ---
@app.route('/airnoise', methods=['POST'])
def predict_airnoise():
    try:
        data = request.get_json()

        # Extract models and label encoders
        aq_model = airnoise_model['aq_model']
        noise_model = airnoise_model['noise_model']
        le_aq = airnoise_model['le_aq']
        le_noise = airnoise_model['le_noise']
        feat_aq = airnoise_model['feat_aq']
        feat_noise = airnoise_model['feat_noise']

        # Prepare input dict matching the feature names exactly
        input_dict = {}

        for col in feat_aq:
            # Map expected column names to your JSON keys
            if col == 'Temperature (°C)':
                input_dict[col] = [float(data['Temperature'])]
            elif col == 'Humidity (%)':
                input_dict[col] = [float(data['Humidity'])]
            elif col == 'MQ135 Value':
                input_dict[col] = [float(data['MQ135'])]
            else:
                # For any other features, check if present in JSON
                input_dict[col] = [float(data.get(col, 0))]

        input_df = pd.DataFrame(input_dict)

        # Predict Air Quality
        aq_pred_idx = aq_model.predict(input_df[feat_aq])[0]

        # Predict Noise — match feature names too
        input_dict_noise = {}
        for col in feat_noise:
            if col == 'Noise_dB':
                input_dict_noise[col] = [float(data['Noise_dB'])]
            else:
                input_dict_noise[col] = [float(data.get(col, 0))]

        input_df_noise = pd.DataFrame(input_dict_noise)
        noise_pred_idx = noise_model.predict(input_df_noise[feat_noise])[0]

        # Decode numeric predictions to labels
        aq_label = le_aq.inverse_transform([aq_pred_idx])[0]
        noise_label = le_noise.inverse_transform([noise_pred_idx])[0]

        return jsonify({
            'Predicted_Air_Quality_Label': aq_label,
            'Predicted_Noise_Label': noise_label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
