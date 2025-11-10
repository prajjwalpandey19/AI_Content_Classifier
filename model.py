# model.py - builder used by train_full.py (optional)
from tensorflow.keras import layers, models

def build_model(num_classes, max_tokens=30000, seq_len=384, embed_dim=192):
    vectorize = layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=seq_len)
    inputs = layers.Input(shape=(1,), dtype="string")
    x = vectorize(inputs)
    x = layers.Embedding(max_tokens, embed_dim, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model, vectorize
