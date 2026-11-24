# Deploy to Hugging Face Spaces
1. Create a new **Gradio** Space.
2. Unzip this archive locally.
3. Copy `best.pt` into `frozen/yolo_detector/best.pt`.
4. Upload/push the **contents of `huggingface_music_annotation_space/` to the Space repository root**. `app.py` and `README.md` must be at repository root.
5. For a private model/application, use a private Space. Large checkpoints should normally be stored with Git LFS.
6. If you want GPU-backed YOLO inference, select a GPU Space hardware tier. CPU fallback exists in the code, but it will be much slower and is not an exact T4 runtime replay.
7. Persistent mutable HITL state needs Hugging Face persistent storage or external object storage before the full reviewer is enabled.
