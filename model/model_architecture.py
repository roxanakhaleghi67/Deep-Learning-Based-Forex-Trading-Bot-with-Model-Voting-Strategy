from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.layers import Bidirectional, GRU, MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model
from keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
import tensorflow as tf

def build_classifier(input_shape, params):
    inputs = Input(shape=input_shape)
    x = Conv1D(params['filters'], kernel_size=params['kernel_size'], activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(params['dropout_rate2'])(x)

    x = Bidirectional(GRU(params['units3'], return_sequences=True, dropout=params['gru_dropout1']))(x)
    x = MultiHeadAttention(num_heads=params['num_heads'], key_dim=params['key_dim'])(x, x)
    x = LayerNormalization()(x)
    x = Dropout(params['dropout_rate3'])(x)

    x = Bidirectional(GRU(params['units3'], return_sequences=False, dropout=params['gru_dropout1']))(x)
    x = Dense(params['units5'], activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(params['dropout_rate5'])(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'], amsgrad=True)
    model.compile(loss=BinaryFocalCrossentropy(gamma=2),
                  optimizer=optimizer,
                  metrics=["accuracy", AUC(name="auc"), Precision(name="precision"), Recall(name="recall")])
    return model
