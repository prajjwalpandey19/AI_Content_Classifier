# utils.py - helper utilities
import pandas as pd

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
        if 2 <= nunique <= 200:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]
    return text_col, label_col
