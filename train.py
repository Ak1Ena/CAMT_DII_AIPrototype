"""
train.py — Retrain the Chihuahua vs Muffin classifier
======================================================
Place images in:
  data/chihuahua/  ← chihuahua images
  data/muffin/     ← muffin images

Run:
  python train.py
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

IMG_SIZE = (128, 128)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLASSES  = ["chihuahua", "muffin"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_image(path):
    return np.array(Image.open(path).convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0


def hog_features(img_arr):
    img = Image.fromarray((img_arr * 255).astype(np.uint8)).convert("L")
    arr = np.array(img, dtype=np.float32)
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi % 180
    cell, feats = 16, []
    for r in range(0, arr.shape[0], cell):
        for c in range(0, arr.shape[1], cell):
            hist, _ = np.histogram(ang[r:r+cell, c:c+cell], bins=9, range=(0, 180),
                                   weights=mag[r:r+cell, c:c+cell])
            feats.extend(hist)
    return np.array(feats)


def color_histogram(img_arr, bins=32):
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(img_arr[:, :, ch], bins=bins, range=(0, 1))
        feats.extend(hist / (hist.sum() + 1e-8))
    return np.array(feats)


def extract_features(img_arr):
    return np.concatenate([hog_features(img_arr), color_histogram(img_arr)])


def augment(img_arr):
    imgs = [img_arr]
    pil = Image.fromarray((img_arr * 255).astype(np.uint8))
    imgs.append(np.array(ImageOps.mirror(pil)) / 255.0)
    for f in [0.7, 1.3]:
        imgs.append(np.array(ImageEnhance.Brightness(pil).enhance(f)) / 255.0)
    imgs.append(np.array(ImageEnhance.Contrast(pil).enhance(1.4)) / 255.0)
    for deg in [-15, 15]:
        imgs.append(np.array(pil.rotate(deg)) / 255.0)
    imgs.append(np.array(pil.filter(ImageFilter.GaussianBlur(1))) / 255.0)
    w, h = pil.size
    imgs.append(np.array(pil.crop((w//8, h//8, w-w//8, h-h//8)).resize((w, h))) / 255.0)
    return imgs


def main():
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"[!] Missing directory: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir) if not f.startswith('.')]
        print(f"[{cls}] {len(files)} base images → augmenting...")
        for fname in files:
            try:
                arr = load_image(os.path.join(cls_dir, fname))
                for aug in augment(arr):
                    X.append(extract_features(aug))
                    y.append(label)
            except Exception as e:
                print(f"  Skip {fname}: {e}")
        print(f"  → {y.count(label)} samples")

    X, y = np.array(X), np.array(y)
    print(f"\nTotal: {len(X)} samples, distribution: {np.bincount(y)}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    print(f"CV accuracy: {scores.mean():.2%} ± {scores.std():.2%}")

    pipeline.fit(X, y)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({'pipeline': pipeline, 'classes': CLASSES, 'img_size': IMG_SIZE}, f)
    print(f"\n✓ Model saved → {OUT_PATH}")


if __name__ == "__main__":
    main()