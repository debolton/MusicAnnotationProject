import gradio as gr

def detect(image):
    # Replace with YOLO inference
    return image

def save_annotations():
    return "Annotations saved"

def train_model():
    return "Training started"

with gr.Blocks(title="Sheet Music AI") as app:

    gr.Markdown("# 🎼 Sheet Music Handwriting Detection")

    with gr.Tab("Detect"):

        image_input = gr.Image(type="filepath")

        detect_btn = gr.Button("Run Detection")

        image_output = gr.Image()

        detect_btn.click(
            fn=detect,
            inputs=image_input,
            outputs=image_output
        )

    with gr.Tab("Review"):

        gr.Markdown(
            "Correct bounding boxes and add missing annotations"
        )

        review_image = gr.Image()

        save_btn = gr.Button("Save Labels")

        save_status = gr.Textbox()

        save_btn.click(
            fn=save_annotations,
            outputs=save_status
        )

    with gr.Tab("Train"):

        train_btn = gr.Button("Train New Model")

        train_output = gr.Textbox()

        train_btn.click(
            fn=train_model,
            outputs=train_output
        )

app.launch()
