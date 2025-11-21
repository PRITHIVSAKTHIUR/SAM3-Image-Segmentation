import os
import spaces
import gradio as gr
import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
from transformers import Sam3Processor, Sam3Model

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    print("Loading SAM3 Model and Processor...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have the correct libraries installed and access to the model.")
    # Fallback/Placeholder for demonstration if model doesn't exist in environment yet
    model = None 
    processor = None

@spaces.GPU
def segment_image(input_image, text_prompt, threshold=0.5):
    if input_image is None:
        raise gr.Error("Please upload an image.")
    if not text_prompt:
        raise gr.Error("Please enter a text prompt (e.g., 'cat', 'face').")
    
    if model is None or processor is None:
        raise gr.Error("Model not loaded correctly.")

    # Convert image to RGB
    image_pil = input_image.convert("RGB")

    # Preprocess
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results['masks'] # Boolean tensor [N, H, W]
    scores = results['scores']
    
    # Prepare for Gradio AnnotatedImage
    # Gradio expects (image, [(mask, label), ...])
    
    annotations = []
    masks_np = masks.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    for i, mask in enumerate(masks_np):
        # mask is a boolean array (True/False). 
        # AnnotatedImage handles the coloring automatically.
        # We just pass the mask and a label.
        score_val = scores_np[i]
        label = f"{text_prompt} ({score_val:.2f})"
        annotations.append((mask, label))
    
    # Return tuple format for AnnotatedImage
    return (image_pil, annotations)

css="""
#col-container {
    margin: 0 auto;
    max-width: 980px;
}
#main-title h1 {font-size: 2.1em !important;}
"""

with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            "# **SAM3 Image Segmentation**", 
            elem_id="main-title"
        )
        
        gr.Markdown("Segment objects in images using **SAM3** (Segment Anything Model 3) with text prompts.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="pil", height=300)
                text_prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="e.g., cat, ear, car wheel...",
                )
                
                run_button = gr.Button("Segment", variant="primary")

            with gr.Column(scale=1.5):
                output_image = gr.AnnotatedImage(label="Segmented Output", height=380)
                
                with gr.Row():
                    threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, value=0.4, step=0.05)
        
        gr.Examples(
            examples=[
                ["examples/player.jpg", "player in white", 0.5],
                ["examples/goldencat.webp", "black cat", 0.4],
                ["examples/taxi.jpg", "blue taxi", 0.5],
            ],
            inputs=[input_image, text_prompt, threshold],
            outputs=[output_image],
            fn=segment_image,
            cache_examples="lazy",
            label="Examples"
        )

    run_button.click(
        fn=segment_image,
        inputs=[input_image, text_prompt, threshold],
        outputs=[output_image]
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, ssr_mode=False, show_error=True)