from pathlib import Path
from music_annotation.hitl_generator import generate_all
ROOT=Path('/content/drive/MyDrive/MusicAnnotationProject')
print(generate_all(ROOT/'source'/'annotated_only',ROOT/'frozen'/'yolo_detector'/'best.pt',ROOT/'human_in_the_middle'/'draft_masks',ROOT/'human_in_the_middle'/'confidence_maps',ROOT/'human_in_the_middle'/'draft_audit',ROOT/'human_in_the_middle'/'audit'))
