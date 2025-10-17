from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# ------------------------------
# Load models once at startup
# ------------------------------
try:
    solubility_model = joblib.load('solubility_model.pkl')
except Exception:
    solubility_model = None

# Baseline RF bundle you already had
try:
    airnoise_rf = joblib.load('air-noise.pkl')  # RandomForest bundle
except Exception:
    airnoise_rf = None

# New: linear (Logistic Regression) bundle
try:
    airnoise_linear = joblib.load('air-noise-linear.pkl')
except Exception:
    airnoise_linear = None

# New: SVM (RBF) bundle
try:
    airnoise_svm = joblib.load('air-noise-svm.pkl')
except Exception:
    airnoise_svm = None


# ------------------------------
# Helpers
# ------------------------------
def _safe_float(d, key, default=None, aliases=()):
    """Read float value from dict using key or alias list."""
    if key in d:
        return float(d[key])
    for a in aliases:
        if a in d:
            return float(d[a])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing key: {key} (aliases: {aliases})")

def _predict_airnoise_with_bundle(bundle, payload):
    """
    bundle: joblib dict with keys:
        aq_model, noise_model, le_aq, le_noise, feat_aq, feat_noise
    payload: dict from request JSON
    """
    if bundle is None:
        raise RuntimeError("Requested model is not loaded. Ensure the .pkl exists next to app.py")

    aq_model = bundle['aq_model']
    noise_model = bundle['noise_model']
    le_aq = bundle['le_aq']
    le_noise = bundle['le_noise']
    feat_aq = bundle['feat_aq']          # e.g. ["Temperature (°C)", "Humidity (%)", "MQ135 Value"]
    feat_noise = bundle['feat_noise']    # e.g. ["Noise (dB)"]

    # Build input for AQ features
    row_aq = []
    for col in feat_aq:
        if col == 'Temperature (°C)':
            row_aq.append(_safe_float(payload, 'Temperature', aliases=('temp', 'Temp', 'temperature')))
        elif col == 'Humidity (%)':
            row_aq.append(_safe_float(payload, 'Humidity', aliases=('hum', 'Hum', 'humidity')))
        elif col == 'MQ135 Value':
            row_aq.append(_safe_float(payload, 'MQ135', aliases=('mq135', 'mq')))
        else:
            # any other feature names present in bundle
            row_aq.append(_safe_float(payload, col, default=0.0))
    X_aq = pd.DataFrame([row_aq], columns=feat_aq)

    # Build input for Noise features
    row_noise = []
    for col in feat_noise:
        if col == 'Noise (dB)':
            row_noise.append(_safe_float(payload, 'Noise_dB', aliases=('noise', 'Noise', 'noise_db')))
        else:
            row_noise.append(_safe_float(payload, col, default=0.0))
    X_noise = pd.DataFrame([row_noise], columns=feat_noise)

    # Predict indices
    aq_idx = aq_model.predict(X_aq)[0]
    n_idx = noise_model.predict(X_noise)[0]

    # Decode to labels
    aq_label = le_aq.inverse_transform([aq_idx])[0]
    noise_label = le_noise.inverse_transform([n_idx])[0]

    # Probabilities if supported (LogReg/SVM with probability=True)
    aq_probs = None
    noise_probs = None
    if hasattr(aq_model, "predict_proba"):
        probs = aq_model.predict_proba(X_aq)[0]
        aq_probs = {cls: float(p) for cls, p in zip(le_aq.classes_, probs)}
    if hasattr(noise_model, "predict_proba"):
        probs = noise_model.predict_proba(X_noise)[0]
        noise_probs = {cls: float(p) for cls, p in zip(le_noise.classes_, probs)}

    return {
        "Predicted_Air_Quality_Label": aq_label,
        "Predicted_Noise_Label": noise_label,
        "Air_Quality_Probabilities": aq_probs,
        "Noise_Probabilities": noise_probs
    }


# ------------------------------
# Healthcheck
# ------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "ok": True,
        "solubility_model": solubility_model is not None,
        "airnoise_rf": airnoise_rf is not None,
        "airnoise_linear": airnoise_linear is not None,
        "airnoise_svm": airnoise_svm is not None
    })


# ------------------------------
# Original Solubility Endpoint
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Solubility prediction endpoint
    Example:
    {
        "MolLogP": 1.2,
        "MolWt": 250.3,
        "NumRotatableBonds": 3,
        "AromaticProportion": 0.25
    }
    """
    if solubility_model is None:
        return jsonify({'error': 'solubility_model.pkl not loaded'}), 500
    try:
        data = request.get_json(force=True)
        X = pd.DataFrame([[
            float(data['MolLogP']),
            float(data['MolWt']),
            int(data['NumRotatableBonds']),
            float(data['AromaticProportion'])
        ]], columns=['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion'])
        pred = solubility_model.predict(X)[0]
        return jsonify({'logS': float(pred)})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


# ------------------------------
# Existing Air & Noise (RF bundle)
# ------------------------------
@app.route('/airnoise', methods=['POST'])
def predict_airnoise_rf():
    """
    Uses the baseline RandomForest bundle: air-noise.pkl
    """
    try:
        data = request.get_json(force=True)
        out = _predict_airnoise_with_bundle(airnoise_rf, data)
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


# ------------------------------
# NEW: Linear model endpoint (Logistic Regression)
# ------------------------------
@app.route('/air-voice-linear', methods=['POST'])
def air_voice_linear():
    """
    Uses: air-noise-linear.pkl
    JSON body keys accepted (any casing):
      - Temperature / temp
      - Humidity / hum
      - MQ135 / mq135
      - Noise_dB / noise / Noise
    """
    try:
        data = request.get_json(force=True)
        out = _predict_airnoise_with_bundle(airnoise_linear, data)
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


# ------------------------------
# NEW: SVM model endpoint (RBF)
# ------------------------------
@app.route('/air-voice-svm', methods=['POST'])
def air_voice_svm():
    """
    Uses: air-noise-svm.pkl
    Same JSON body format as /air-voice-linear
    """
    try:
        data = request.get_json(force=True)
        out = _predict_airnoise_with_bundle(airnoise_svm, data)
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400


if __name__ == '__main__':
    # Set threaded=True if you expect concurrent requests on a small server
    app.run(host='0.0.0.0', port=10000, debug=False, threaded=True)
