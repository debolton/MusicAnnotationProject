"""Gradio 6 V12 launcher. Browser HTML/JS/CSS are loaded from package assets."""
from pathlib import Path
ASSET_DIR=Path(__file__).parent/'assets'
def asset(name):return (ASSET_DIR/name).read_text()
def launch(*args,**kwargs):
    raise RuntimeError('Use examples/07_thin_launcher.py; it wires the exact Notebook 07 backend to the packaged V12 browser assets.')
