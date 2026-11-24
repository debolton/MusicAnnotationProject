# HF-02.4 direct-event patch
Replace the HF-02.3 root `app.py` with this file. Keep all other HF-02.3 files and frozen artefacts.

The load path is now `generate Python -> page_package output -> .then(js=LOAD_JS, inputs=page_package)`.
The return path is `Capture reviewed mask -> js=EXPORT_JS -> add_mask/erase_mask outputs -> Authorise reviewed mask Python`.
No MutationObserver and no JavaScript query of Gradio textbox DOM is used by app.py.

For the canvas module, `window.HITLReviewer.loadPage(package)` and `window.HITLReviewer.exportMasks()` must be exposed. If the current HF-02.3 `notebook07_canvas.py` does not expose those methods, use the included canvas replacement in the full package rather than this one-file patch.
