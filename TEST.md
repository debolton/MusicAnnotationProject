# iPad/iPhone test
1. Upload Page 28 and click **Generate standard-Gradio diagnostic**.
2. Open **Standard images** and confirm all four images display.
3. Open **Standard ImageEditor** and confirm the synthetic mask appears.
4. Draw one stroke with the editor's own tools.
5. Click **Read ImageEditor value** and confirm the returned composite displays.

Interpretation:
- Standard images fail: capture the Space logs and screenshot. The failure is below the custom reviewer layer.
- Standard images work but ImageEditor fails: standard `gr.ImageEditor` on iOS is the isolated limitation.
- Both work: reconnect YOLO first, then mask persistence, then canonical V7.8/V7.7F.
