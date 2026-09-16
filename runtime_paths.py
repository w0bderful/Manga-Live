"""Keep portable user data beside the executable and bundled assets inside it."""
from pathlib import Path
import os
import sys

RESOURCE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else RESOURCE_DIR
if getattr(sys, 'frozen', False) and os.environ.get('MANGA_LIVE_DATA_DIR'):
    APP_DIR = Path(os.environ['MANGA_LIVE_DATA_DIR']).resolve()
