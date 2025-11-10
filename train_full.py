#!/usr/bin/env python3
"""train_full.py
Full-accuracy training pipeline for ContentSense.

Usage examples:
  # Run with defaults (expects CSV in --csv)
  python train_full.py --csv /path/to/ai_human_content_detection_dataset.csv --epochs 50 --batch_size 64 --save_dir saved_model

  # On Google Colab use a GPU runtime and increase batch_size, e.g.
  python train_full.py --csv /content/ai_human_content_detection_dataset.csv --epochs 50 --batch_size 64 --save_dir saved_model
"""

import argparse, os, json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def detect_text_and_label_columns(df):
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if text_cols:
        avg_len = {c: df[c].dropna().astype(str).map(len).mean() for c in text_cols}
        text_col = max(avg_len, key=avg_len.get)
    else:
        text_col = df.columns[0]
    label_col = None
    for c in df.columns:
        if c == text_col: continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= 200:  # allow many classes if needed
            label_col = c
            break
    if label_col is None:
        for c in df.columns:
            if c.lower() in ("label","labels","target","class","y"):
                label_col = c
                break
    if label_col is None:
        label_col = df.columns[-1]
    return text_col, label_col

def build_model(num_classes, max_tokens=50000, seq_len=512, embed_dim=192):
    vectorize = layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=seq_len)
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorize(inputs)
    x = layers.Embedding(max_tokens, embed_dim, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=metrics)
    return model, vectorize

def df_to_ds(df, text_col, label_col, batch_size=32, shuffle=True):
    texts = df[text_col].values
    labels = df[label_col].values
    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main(args):
    print("Loading CSV:", args.csv)
    df = pd.read_csv(args.csv)
    print("Shape:", df.shape)
    text_col, label_col = detect_text_and_label_columns(df)
    print("Detected text column:", text_col, "label column:", label_col)
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str)
    le = LabelEncoder()
    df["_label_enc"] = le.fit_transform(df[label_col].astype(str))
    num_classes = len(le.classes_)
    print("Num classes:", num_classes)

    train_df, val_df = train_test_split(df, test_size=args.val_split, stratify=df["_label_enc"], random_state=42)
    train_df = train_df.rename(columns={"_label_enc":"label"})
    val_df = val_df.rename(columns={"_label_enc":"label"})

    model, vectorize = build_model(num_classes=num_classes, max_tokens=args.max_tokens, seq_len=args.seq_len, embed_dim=args.embed_dim)
    # adapt vectorize on training texts
    print("Adapting text vectorization...")
    text_ds = tf.data.Dataset.from_tensor_slices(train_df[text_col].values).batch(args.batch_size)
    vectorize.adapt(text_ds)

    # prepare datasets
    train_ds = df_to_ds(train_df, text_col, "label", batch_size=args.batch_size, shuffle=True)
    val_ds = df_to_ds(val_df, text_col, "label", batch_size=args.batch_size, shuffle=False)

    # callbacks: checkpointing, early stopping, reduceLR, tensorboard
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    cb_checkpoint = callbacks.ModelCheckpoint(os.path.join(save_dir, "best_model.h5"), save_best_only=True, monitor="val_loss", mode="min")
    cb_early = callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True)
    cb_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
    tb_logdir = os.path.join(save_dir, "tensorboard")
    cb_tb = callbacks.TensorBoard(log_dir=tb_logdir, histogram_freq=1)

    print("Starting training for", args.epochs, "epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[cb_checkpoint, cb_early, cb_reduce, cb_tb])

    # save final model (SavedModel) and label classes
    final_model_dir = os.path.join(save_dir, "saved_model")
    model.save(final_model_dir, include_optimizer=False)
    with open(os.path.join(final_model_dir, "label_classes.json"), "w", encoding="utf8") as f:
        json.dump(list(le.classes_), f, ensure_ascii=False)

    # save history and metadata
    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf8") as f:
        json.dump(history.history, f, ensure_ascii=False)
    print("Training complete. Saved to:", save_dir)
    print("To deploy, copy the 'saved_model' folder and the serve.py file into your deployment repo.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV file")
    p.add_argument("--save_dir", default="experiment_outputs", help="Where to save model + logs")
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs for full accuracy")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--max_tokens", type=int, default=30000)
    p.add_argument("--seq_len", type=int, default=384)
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--early_stopping_patience", type=int, default=6)
    args = p.parse_args()
    main(args)
