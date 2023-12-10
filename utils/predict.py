import numpy as np
import pickle

# Predicting
with open("model/xgboost-v.1.0.0.pkl", "rb") as xgb:
    model = pickle.load(xgb)

def predict(X):
    return np.round(model.predict(X))