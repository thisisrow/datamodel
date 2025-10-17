import joblib
model = joblib.load('air-noise.pkl')
print(type(model))

# If it's a dict, show what keys it has
if isinstance(model, dict):
    print(model.keys())
