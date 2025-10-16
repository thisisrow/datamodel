import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import sklearn

print(f"scikit-learn version: {sklearn.__version__}")  # Should print 1.4.2

# Load dataset
df = pd.read_csv('delaney_solubility_with_descriptors.csv')
X = df[['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion']]
y = df['logS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save model
joblib.dump(model, 'solubility_model.pkl')