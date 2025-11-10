# ContentSense — Full Accuracy Project

This folder contains a full training pipeline and deployment files for ContentSense.
It is configured for "full accuracy" training: higher epochs, larger model, checkpoints and TensorBoard logs.

## Files
- `train_full.py`    — full training script (use GPU for best results)
- `serve.py`         — Flask API to serve SavedModel
- `model.py`         — optional model builder
- `utils.py`         — helper functions
- `requirements.txt` — dependencies
- `Procfile`         — for Heroku/Render (gunicorn)
- `train_notes.txt`  — tips for achieving best accuracy

## Recommended (Google Colab)
1. Open Google Colab and choose **Runtime > Change runtime type > GPU** (preferably TPU if you change code for it).
2. Upload your CSV (e.g. `ai_human_content_detection_dataset.csv`).
3. Run training:
   ```bash
   python train_full.py --csv /content/ai_human_content_detection_dataset.csv --epochs 50 --batch_size 64 --save_dir /content/outputs
   ```
4. After training, the best model is saved under `/content/outputs/<timestamp>/saved_model`.
5. Download or copy `saved_model` and `label_classes.json` to your deployment repo and use `serve.py` to serve.

## Tips for full accuracy
- Use GPU runtime with at least 12GB RAM (Colab Pro recommended).
- Increase `epochs` (30-100) and `batch_size` depending on GPU memory.
- Monitor TensorBoard logs (use `tensorboard --logdir /path/to/outputs/<timestamp>/tensorboard`).
- Consider switching to a transformer model (Hugging Face) for large datasets and improved performance.