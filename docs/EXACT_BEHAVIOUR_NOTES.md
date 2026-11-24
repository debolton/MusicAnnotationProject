# Exact behaviour notes

Notebook 06 extraction is source-faithful: 640 tiles, 320 stride, padded white canvas, max confidence merge, nearest-neighbour mask resize, >0.5 instance mask, float16 confidence `.npy`, binary draft at 0.20, and the original reviewed/current skip rules.

Notebook 07 backend extraction is source-faithful: queue by source MD5, sequential `PageNNN` naming, unchanged source copied to `aligned`, corrected mask `(draft OR ADD) AND NOT ERASE`, 60/40 red preview, and the original JSON audit schema.

The browser-side UI remains presentation code rather than model/backend code. The exact supplied Notebook 07 should be retained as the canonical UI source. Its Python backend has been moved into the library; a later notebook conversion can import that backend while keeping the existing HTML/JS/CSS cell unchanged.
