import numpy as np
from sklearn.preprocessing import RobustScaler

def scale_features(X_train, X_test):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def X_3d_RNN(X, y, lag):
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        print("NaN in data!")
        return None, None

    X_3d, y_3d = [], []
    for i in range(lag, len(X)):
        X_3d.append(X[i - lag:i])
        y_3d.append(y[i])
    return np.array(X_3d), np.array(y_3d)
