from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('solubility_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([[
            float(data['MolLogP']),
            float(data['MolWt']),
            int(data['NumRotatableBonds']),
            float(data['AromaticProportion'])
        ]], columns=['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion'])
        prediction = model.predict(input_data)[0]
        return jsonify({'logS': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)