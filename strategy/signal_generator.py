import numpy as np

def generate_signal_from_model_output(pred, threshold=0.5):
    # pred: np.array of predicted probabilities
    if pred < threshold:
        return -1  # sell
    elif pred > 1 - threshold:
        return 1   # buy
    else:
        return 0   # no-trade
