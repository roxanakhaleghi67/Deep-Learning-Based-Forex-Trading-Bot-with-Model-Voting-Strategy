import numpy as np
from sklearn.metrics import f1_score

def evaluate_model(model, X_test, y_test, lag=35, threshold=0.5):
    y_pred = model.predict(X_test)
    y_pred_all = np.concatenate((np.zeros([lag, 1]), y_pred), axis=0).flatten()
    y_true_all = y_test.flatten()

    mask = y_pred_all != 0
    pred_binary = (y_pred_all[mask] > threshold).astype(int)
    true_binary = ((y_true_all[mask] > 0.5)).astype(int)

    f1 = f1_score(true_binary, pred_binary)
    return f1, pred_binary, true_binary
