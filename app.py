import cv2
import gradio as gr
import numpy as np

MAX_WIDTH = 960

def as_rgb(image):
    a = np.asarray(image)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
    return a

def fit(image, max_width=MAX_WIDTH):
    image = as_rgb(image)
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    nh = max(1, round(h * max_width / w))
    return cv2.resize(image, (max_width, nh), interpolation=cv2.INTER_AREA)

def synthetic_draft(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray < 135, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

def overlay(rgb, mask):
    colour = rgb.copy()
    colour[mask > 0] = (255, 45, 45)
    return cv2.addWeighted(rgb, 0.62, colour, 0.38, 0)

def editor_value(background):
    return {"background": background, "layers": [], "composite": background.copy()}

def prepare(image):
    if image is None:
        raise gr.Error("Upload a page first.")
    rgb = fit(image)
    mask = synthetic_draft(rgb)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    confidence = np.zeros_like(mask)
    confidence[mask > 0] = 210
    confidence_rgb = cv2.cvtColor(cv2.applyColorMap(confidence, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return rgb, overlay(rgb, mask), mask_rgb, confidence_rgb, editor_value(mask_rgb), f"Diagnostic images created at {rgb.shape[1]} x {rgb.shape[0]}. No YOLO, V7.8 or V7.7F ran."

def inspect_editor(value):
    if value is None:
        return None, "No ImageEditor value received."
    if not isinstance(value, dict):
        a = as_rgb(value)
        return a, f"ImageEditor returned {type(value).__name__}, shape {a.shape}."
    comp = value.get("composite")
    bg = value.get("background")
    layers = value.get("layers") or []
    chosen = comp if comp is not None else bg
    if chosen is None:
        return None, f"ImageEditor dictionary received, but no composite/background. Keys: {sorted(value)}"
    a = as_rgb(chosen)
    return a, f"ImageEditor dictionary received. Composite shape {a.shape}; layers {len(layers)}; keys {sorted(value)}."

with gr.Blocks(title="HF-02.9 Standard Gradio Diagnostic") as demo:
    gr.Markdown("""# HF-02.9 Standard Gradio Diagnostic
This build uses only standard `gr.Image` and `gr.ImageEditor` components. It contains no custom HTML, JavaScript, hidden browser bridge, YOLO, V7.8, V7.7F or component workflow.""")
    with gr.Tab("1 Upload"):
        upload = gr.Image(type="numpy", label="Upload annotated page")
        generate = gr.Button("Generate standard-Gradio diagnostic", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
    with gr.Tab("2 Standard images"):
        original = gr.Image(type="numpy", label="Original preview", height=520)
        diagnostic_overlay = gr.Image(type="numpy", label="Synthetic diagnostic overlay", height=520)
        mask = gr.Image(type="numpy", label="Synthetic mask only", height=520)
        confidence = gr.Image(type="numpy", label="Synthetic confidence view", height=520)
    with gr.Tab("3 Standard ImageEditor"):
        editor = gr.ImageEditor(type="numpy", label="Standard Gradio ImageEditor diagnostic", height=700)
        inspect = gr.Button("Read ImageEditor value", variant="primary")
        editor_output = gr.Image(type="numpy", label="Composite returned to Python", height=520)
        editor_status = gr.Textbox(label="Editor return diagnostics", interactive=False)
    generate.click(prepare, upload, [original, diagnostic_overlay, mask, confidence, editor, status], show_progress="full")
    inspect.click(inspect_editor, editor, [editor_output, editor_status], show_progress="minimal")

if __name__ == "__main__":
    demo.launch()
