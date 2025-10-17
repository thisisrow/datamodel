#!/usr/bin/env python3
# airnoise-svm.py
# Train SVM (RBF) models for Air Quality Label & Noise Label

import argparse, joblib, sklearn, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_eval(csv_path: str, save_path: str, test_size: float = 0.20, random_state: int = 42):
    print(f"scikit-learn: {sklearn.__version__}")

    df = pd.read_csv(csv_path)
    req = ["Temperature (°C)", "Humidity (%)", "MQ135 Value", "Air Quality Label",
           "Noise (dB)", "Noise Label"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    feat_aq = ["Temperature (°C)", "Humidity (%)", "MQ135 Value"]
    feat_noise = ["Noise (dB)"]
    y_aq_raw = df["Air Quality Label"].astype(str)
    y_noise_raw = df["Noise Label"].astype(str)

    le_aq, le_noise = LabelEncoder(), LabelEncoder()
    y_aq = le_aq.fit_transform(y_aq_raw)
    y_noise = le_noise.fit_transform(y_noise_raw)

    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        df[feat_aq], y_aq, test_size=test_size, random_state=random_state, stratify=y_aq
    )
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        df[feat_noise], y_noise, test_size=test_size, random_state=random_state, stratify=y_noise
    )

    # Pipelines (scaler + SVM-RBF)
    aq_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state))
    ])
    noise_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state))
    ])

    aq_pipe.fit(Xa_tr, ya_tr)
    noise_pipe.fit(Xn_tr, yn_tr)

    def report(name, y_true, y_pred, classes):
        print(f"\n=== {name} ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print("Classes:", list(classes))
        print(classification_report(y_true, y_pred, target_names=classes))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    y_aq_pred = aq_pipe.predict(Xa_te)
    y_n_pred  = noise_pipe.predict(Xn_te)

    report("Air Quality Label (SVM-RBF)", ya_te, y_aq_pred, le_aq.classes_)
    report("Noise Label (SVM-RBF)",       yn_te, y_n_pred,  le_noise.classes_)

    bundle = {
        "aq_model": aq_pipe,
        "noise_model": noise_pipe,
        "le_aq": le_aq,
        "le_noise": le_noise,
        "feat_aq": feat_aq,
        "feat_noise": feat_noise,
        "sklearn_version": sklearn.__version__,
        "model_type": "svm_rbf"
    }
    joblib.dump(bundle, save_path)
    print(f"\nSaved bundle -> {save_path}")

def predict_with_bundle(pkl, temp, hum, mq135, noise):
    b = joblib.load(pkl)
    import pandas as pd
    Xa = pd.DataFrame([[temp, hum, mq135]], columns=b["feat_aq"])
    Xn = pd.DataFrame([[noise]], columns=b["feat_noise"])
    aq_code = b["aq_model"].predict(Xa)[0]
    n_code  = b["noise_model"].predict(Xn)[0]
    aq = b["le_aq"].inverse_transform([aq_code])[0]
    nl = b["le_noise"].inverse_transform([n_code])[0]
    return aq, nl

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sensor_dataset.csv")
    ap.add_argument("--save", default="air-noise-svm.pkl")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--predict", action="store_true", help="Run a one-shot prediction instead of training")
    ap.add_argument("--temp", type=float)
    ap.add_argument("--hum", type=float)
    ap.add_argument("--mq135", type=float)
    ap.add_argument("--noise", type=float)
    args = ap.parse_args()

    if args.predict:
        if None in (args.temp, args.hum, args.mq135, args.noise):
            raise SystemExit("Provide --temp --hum --mq135 --noise with --predict")
        aq, nl = predict_with_bundle(args.save, args.temp, args.hum, args.mq135, args.noise)
        print(f"Predicted AQ Label: {aq}")
        print(f"Predicted Noise Label: {nl}")
    else:
        train_and_eval(args.csv, args.save, args.test_size)
