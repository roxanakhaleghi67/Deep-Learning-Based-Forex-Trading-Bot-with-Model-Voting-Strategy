import xgboost as xgb
import numpy as np

def select_features(X, y, top_k=15):
    model = xgb.XGBClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    selected = X.columns[np.argsort(importances)[-top_k:]]
    return X[selected]
