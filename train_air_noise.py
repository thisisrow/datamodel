# train_air_noise.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sklearn
import numpy as np

print(f"scikit-learn version: {sklearn.__version__}")  # expect 1.4.x

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("sensor_dataset.csv")

# Basic sanity checks
required_cols = [
    "Temperature (°C)", "Humidity (%)", "MQ135 Value",
    "Air Quality Label", "Noise (dB)", "Noise Label"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Feature sets / targets
feat_aq   = ["Temperature (°C)", "Humidity (%)", "MQ135 Value"]
target_aq = "Air Quality Label"

# For noise label we mainly need the dB; you can extend to more features if helpful.
feat_noise   = ["Noise (dB)"]
target_noise = "Noise Label"

# -----------------------------
# Encode labels
# -----------------------------
le_aq   = LabelEncoder()
le_noise = LabelEncoder()

y_aq_enc    = le_aq.fit_transform(df[target_aq].astype(str))
y_noise_enc = le_noise.fit_transform(df[target_noise].astype(str))

# Split (80/20)
X_aq_train, X_aq_test, y_aq_train, y_aq_test = train_test_split(
    df[feat_aq], y_aq_enc, test_size=0.20, random_state=42, stratify=y_aq_enc
)
X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(
    df[feat_noise], y_noise_enc, test_size=0.20, random_state=42, stratify=y_noise_enc
)

# -----------------------------
# Train models
# -----------------------------
aq_model = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=1,
    class_weight="balanced", random_state=42, n_jobs=-1
)
noise_model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_leaf=1,
    class_weight="balanced", random_state=42, n_jobs=-1
)

aq_model.fit(X_aq_train, y_aq_train)
noise_model.fit(X_n_train, y_n_train)

# -----------------------------
# Evaluate
# -----------------------------


def show_report(name, y_true, y_pred, le):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classes:", list(le.classes_))
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


y_aq_pred  = aq_model.predict(X_aq_test)
y_n_pred   = noise_model.predict(X_n_test)

show_report("Air Quality Label", y_aq_test, y_aq_pred, le_aq)
show_report("Noise Label",       y_n_test,  y_n_pred,  le_noise)

# -----------------------------
# Save everything
# -----------------------------
bundle = {
    "aq_model": aq_model,
    "noise_model": noise_model,
    "le_aq": le_aq,
    "le_noise": le_noise,
    "feat_aq": feat_aq,
    "feat_noise": feat_noise,
    "sklearn_version": sklearn.__version__,
}
joblib.dump(bundle, "air-noise.pkl")
print("\nSaved model bundle -> air-noise.pkl")

# -----------------------------
# Example: Predict for one sample
# -----------------------------
def predict_labels(temp_c, hum_pct, mq135_value, noise_db, bundle=bundle):
    X_aq = pd.DataFrame([[temp_c, hum_pct, mq135_value]], columns=bundle["feat_aq"])
    X_n  = pd.DataFrame([[noise_db]], columns=bundle["feat_noise"])

    aq_code    = bundle["aq_model"].predict(X_aq)[0]
    noise_code = bundle["noise_model"].predict(X_n)[0]

    aq_label    = bundle["le_aq"].inverse_transform([aq_code])[0]
    noise_label = bundle["le_noise"].inverse_transform([noise_code])[0]
    return aq_label, noise_label

sample = df.sample(1, random_state=0).iloc[0]
pred_aq, pred_noise = predict_labels(
    float(sample["Temperature (°C)"]),
    float(sample["Humidity (%)"]),
    float(sample["MQ135 Value"]),
    float(sample["Noise (dB)"])
)
print("\nExample prediction:")
print("Inputs:",
      dict(Temperature=float(sample["Temperature (°C)"]),
           Humidity=float(sample["Humidity (%)"]),
           MQ135=float(sample["MQ135 Value"]),
           Noise_dB=float(sample["Noise (dB)"])))
print("Predicted AQ Label:", pred_aq)
print("Predicted Noise Label:", pred_noise)
