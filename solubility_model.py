import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load & Prepare Data (replace with your full file path)
df = pd.read_csv('delaney_solubility_with_descriptors.csv')  # Use the attached file

X = df[['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion']]
y = df['logS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2-3. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")  # Example output: ~0.75 (actual may vary)

# 5. Save Model
joblib.dump(model, 'solubility_model.pkl')
print("Model saved as solubility_model.pkl")