"""Separate per-user settings, executable data, and bundled assets."""
from pathlib import Path
import os
import sys

RESOURCE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'Manga Live'
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else RESOURCE_DIR
if getattr(sys, 'frozen', False) and os.environ.get('MANGA_LIVE_DATA_DIR'):
    APP_DIR = Path(os.environ['MANGA_LIVE_DATA_DIR']).resolve()

os.environ.setdefault('HF_HOME', str(APP_DIR / '.models' / 'huggingface'))
