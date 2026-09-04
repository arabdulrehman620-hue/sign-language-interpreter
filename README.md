# Signa — Sign Language Recognition

This project has three independent parts. Pick the one you need.

## Setup (one-time)

Both the web app (section 1) and the Python scripts (section 2) run out of a virtual
environment at `.venv/`. Requires Python 3.9–3.11 (mediapipe and tensorflow don't yet
support 3.12+) — check your version with `python --version`.

From the `sign_language_app` folder, create the venv:

```powershell
cd sign_language_app
python -m venv .venv
```

If your default `python` is 3.12 or newer, point `venv` at an older interpreter instead,
e.g. if you have Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
```

This only creates the environment — see section 2 below to install the actual Python
dependencies (`requirements.txt`) into it.

## 1. Web app (`index.html`)

The page recognizes a hand sign from the webcam using MediaPipe Hands + TensorFlow.js
(both loaded from a CDN, so you need internet access), and loads the trained model from
the local `models/` folder via `fetch()`.

Because it uses `fetch()` for the model files, y
ou **cannot** open `index.html` directly
as a `file://` URL — the browser will block those requests with a CORS error. Serve the
folder over HTTP instead, using either of these:

**Option A — activate the venv first, then use plain commands:**

```powershell
cd sign_language_app
.\.venv\Scripts\activate
python -m http.server 8000
```

(Your prompt will show `(.venv)` once it's activated.)

**Option B — call the venv's Python directly, no activation:**

```powershell
cd sign_language_app
.\.venv\Scripts\python.exe -m http.server 8000
```

Then open http://localhost:8000/index.html in your browser and allow camera access.

Stop the server with `Ctrl+C`.

## 2. Python scripts (`scripts/`)

Used to collect training data, train a new model, and run live recognition from the
webcam with the `.venv` virtual environment. Dependencies aren't installed yet — install
them first, using either of these:

**Option A — activate the venv first, then use plain commands:**

```powershell
cd sign_language_app
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Once activated, every command below can drop the `.\.venv\Scripts\python.exe` prefix and
just use `python`:

```powershell
python scripts\collect_data.py hello --sequences 30
python scripts\train_model.py --data-dir data --model-dir models
python scripts\run_live.py --model models\sign_language_model.keras --labels models\labels.json
```

Deactivate anytime with `deactivate`.

**Option B — call the venv's executables directly, no activation:**

```powershell
cd sign_language_app
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Then, from the `sign_language_app` folder:

- **Collect training data** for a sign (opens your webcam, saves landmark sequences under `data/<action>/`):
  ```powershell
  .\.venv\Scripts\python.exe scripts\collect_data.py hello --sequences 30
  ```
- **Train a model** from everything collected under `data/`:
  ```powershell
  .\.venv\Scripts\python.exe scripts\train_model.py --data-dir data --model-dir models
  ```
- **Run live recognition** with a trained Keras model (speaks the recognized word out loud):
  ```powershell
  .\.venv\Scripts\python.exe scripts\run_live.py --model models\sign_language_model.keras --labels models\labels.json
  ```
- **`detect_test.py`** is a smaller manual test script for checking detection without the full pipeline.

Run any script with `-h` to see all options (camera index, sequence length, epochs, confidence threshold, etc).

Note: the model currently in `models/` (`partner_model.json` / `group1-shard1of1.bin`) is
a TensorFlow.js model for the web app, not the `.keras` file `run_live.py` expects by
default — train one with `train_model.py` first, or pass `--model`/`--labels` pointing at
whatever model you have.

## 3. Android app (`android_app/`)

Wraps the web app UI in a native WebView, with the TensorFlow.js model bundled locally
under `app/src/main/assets/models/`. See [android_app/README.md](android_app/README.md)
for full build steps. Summary:

1. Install Android Studio with the Android SDK and SDK Platform 35.
2. Open the `android_app` folder in Android Studio and let Gradle sync.
3. **Build > Build APK(s)**.
4. Install on a connected phone: `adb install -r app\build\outputs\apk\debug\app-debug.apk`

It still needs internet access (TensorFlow.js/MediaPipe load from a CDN) and camera permission.
