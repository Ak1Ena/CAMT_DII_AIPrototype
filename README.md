# The Classifier — Flask web app

A small Flask web application that serves the EfficientNet-B0 image
classifier trained in `efficientnet_b0_image_classification.ipynb`.
The frontend takes an image, sends it to the backend, and displays the
predicted class with its confidence and the top-5 runners-up.

## Project layout

```
flask_app/
├── app.py              # Flask server + REST API
├── model_utils.py      # Loads the .pth checkpoint and runs inference
├── requirements.txt
├── README.md
├── templates/
│   └── index.html      # The single-page UI
└── static/
    ├── style.css       # Editorial / magazine styling
    └── app.js          # Upload, drag-drop, result rendering
```

## Setup

1. **Install dependencies** (Python 3.9+ recommended, ideally inside a venv):

   ```bash
   pip install -r requirements.txt
   ```

   On systems without CUDA, the CPU build of PyTorch is fine — inference for
   a single image on EfficientNet-B0 takes well under a second on CPU.

2. **Place your trained checkpoint.** The notebook saves a file like
   `efficientnet_b0_YYYYMMDD_HHMMSS.pth`. Either:

   - copy/rename it to `model.pth` next to `app.py`, or
   - point the `MODEL_PATH` environment variable at it:

     ```bash
     export MODEL_PATH=/absolute/path/to/efficientnet_b0_20260101_120000.pth
     ```

3. **Run the server:**

   ```bash
   python app.py
   ```

   Open <http://localhost:5000> in your browser.

## How it works

- `model_utils.ImageClassifier` rebuilds an EfficientNet-B0, swaps the final
  FC layer to match `num_classes` from the checkpoint, loads the weights,
  and applies the same normalization used during training.
- `POST /api/predict` accepts a multipart form with an `image` field and
  returns JSON like:

  ```json
  {
    "predicted_class": "chihuahua",
    "confidence": 0.9421,
    "top_predictions": [
      { "class": "chihuahua", "probability": 0.9421 },
      { "class": "muffin",    "probability": 0.0512 }
    ]
  }
  ```

- `GET /api/info` returns the loaded model's class names, image size,
  validation accuracy, and device.

## Notes

- The maximum upload size is 16 MB (configurable in `app.py`).
- The app loads the model lazily; if the checkpoint is missing, the page
  will render with a clear error banner instead of a 500.
- For production, run behind a real WSGI server (e.g. `gunicorn -w 2 app:app`)
  rather than Flask's development server.
