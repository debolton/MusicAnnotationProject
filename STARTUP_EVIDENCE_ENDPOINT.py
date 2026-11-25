# HF-04.4 startup evidence download endpoint
# Merge into app/main.py.
# Required imports:
#   import io, zipfile
#   from pathlib import Path
#   from fastapi import HTTPException
#   from fastapi.responses import StreamingResponse

STARTUP_EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "startup_evidence"

@app.get("/startup-evidence")
def download_startup_evidence():
    root = STARTUP_EVIDENCE_DIR
    required = [
        root / "ardross_v78_equivalence" / "channel_comparison.csv",
        root / "ardross_v78_equivalence" / "summary.json",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Startup evidence is not ready: " + ", ".join(missing),
        )

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise HTTPException(status_code=503, detail="No startup evidence files exist.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=str(path.relative_to(root)))
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="hf_04_4_startup_evidence.zip"',
            "Cache-Control": "no-store",
        },
    )
