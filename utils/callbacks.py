"""Keras callbacks used during training."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from tensorflow import keras


class F1Metric(keras.callbacks.Callback):
    """Compute the validation F1 score at the end of each epoch.

    The score is stored in ``logs["val_f1"]`` so that ``EarlyStopping``,
    ``ModelCheckpoint`` and ``ReduceLROnPlateau`` can monitor it.
    """

    def __init__(self, validation_data, threshold: float = 0.5):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):  # noqa: ARG002 - Keras signature
        logs = logs if logs is not None else {}
        predictions = np.asarray(self.model.predict(self.X_val, verbose=0)).ravel()
        binary = (predictions > self.threshold).astype(int)
        truth = (np.asarray(self.y_val).ravel() > 0.5).astype(int)
        logs["val_f1"] = float(f1_score(truth, binary))