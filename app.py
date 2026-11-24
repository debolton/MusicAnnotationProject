import gradio as gr
from music_annotation.hf_pipeline import analyse_page

with gr.Blocks(title="Music Annotation Removal Assistant") as demo:
    gr.Markdown("""# Music Annotation Removal Assistant
Upload an annotated score page. The Space runs the frozen YOLO draft-mask generator and shows the conservative V7.8 proposal. **This build does not auto-authorise the YOLO mask. Review remains required.**""")
    with gr.Row():
        inp=gr.Image(type="numpy",label="Annotated page")
        overlay=gr.Image(label="YOLO draft-mask overlay")
    with gr.Row():
        heat=gr.Image(label="YOLO confidence heatmap")
        draft=gr.Image(label="Draft mask")
    corrected=gr.Image(label="V7.8 conservative proposal")
    status=gr.Textbox(label="Status",interactive=False)
    run=gr.Button("Generate draft and proposal",variant="primary")
    run.click(analyse_page,inputs=inp,outputs=[overlay,heat,draft,corrected,status])

if __name__=="__main__":
    demo.launch()
