from pathlib import Path

p = Path("app/main.py")
if not p.exists():
    raise SystemExit("Run from the HF-04.4 repository root; app/main.py was not found.")
s = p.read_text()

# HF-04.x main.py already imports io/zipfile in the standard build. Add only if absent.
if "import io" not in s:
    s = "import io\n" + s
if "import zipfile" not in s:
    s = "import zipfile\n" + s
if "StreamingResponse" not in s:
    old = "from fastapi.responses import HTMLResponse,Response,FileResponse"
    if old in s:
        s = s.replace(old, old + ",StreamingResponse")
    else:
        s = "from fastapi.responses import StreamingResponse\n" + s

marker = "# HF-04.4 startup evidence download endpoint"
if marker not in s:
    endpoint = Path("STARTUP_EVIDENCE_ENDPOINT.py").read_text()
    # main.py already owns imports/app. Keep just route block onward.
    route = endpoint[endpoint.index(marker):]
    route = "\n".join(
        line for line in route.splitlines()
        if not line.startswith("# Required imports:")
        and not line.startswith("#   ")
        and not line.startswith("# Merge into")
    )
    s += "\n\n" + route + "\n"

p.write_text(s)
print("[OK] Added GET /startup-evidence to app/main.py")
