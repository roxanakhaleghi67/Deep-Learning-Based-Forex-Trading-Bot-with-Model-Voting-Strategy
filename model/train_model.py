from models.model_architecture import build_rnn_model
from utils.metrics import F1Metric
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(X_train, y_train, X_val, y_val, params, batch_size=128):
    model = build_rnn_model(input_shape=X_train.shape[1:], params=params)

    f1_callback = F1Metric(validation_data=(X_val, y_val))
    callbacks = [
        f1_callback,
        EarlyStopping(monitor='val_f1', mode='max', patience=3, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", monitor='val_f1', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_f1', factor=0.2, patience=2, min_delta=0.001, min_lr=1e-5, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history
