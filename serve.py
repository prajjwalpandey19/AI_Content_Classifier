#!/usr/bin/env python3
# serve.py - Flask API to serve a SavedModel produced by train_full.py
import argparse, os, json
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="saved_model", help="Path to SavedModel directory")
parser.add_argument("--port", type=int, default=5000)
args, _ = parser.parse_known_args()

model = tf.keras.models.load_model(args.model_dir)
# label_classes.json should live next to the SavedModel (same dir)
with open(os.path.join(args.model_dir, "label_classes.json"), "r", encoding="utf8") as f:
    classes = json.load(f)

@app.route("/", methods=["GET"])
def home():
    return {"status":"ok", "message":"ContentSense API running."}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    texts = data.get("texts") or data.get("text") or []
    if isinstance(texts, str):
        texts = [texts]
    preds = model.predict(texts)
    if preds.shape[-1] == 1:
        confs = preds.ravel().tolist()
        labels = [(1 if v>0.5 else 0) for v in confs]
        labels = [classes[i] for i in labels]
        out = [{"text":t, "label":lab, "confidence":float(conf)} for t,lab,conf in zip(texts, labels, confs)]
    else:
        idxs = preds.argmax(axis=1)
        confs = preds.max(axis=1).tolist()
        labels = [classes[int(i)] for i in idxs]
        out = [{"text":t, "label":lab, "confidence":float(conf)} for t,lab,conf in zip(texts, labels, confs)]
    return jsonify({"predictions": out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
