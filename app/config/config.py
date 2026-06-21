from pathlib import Path

# Absolute path to the app/model directory, computed relative to this file.
# This makes model loading work no matter what directory you launch the app from.
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = str(BASE_DIR / "model")
