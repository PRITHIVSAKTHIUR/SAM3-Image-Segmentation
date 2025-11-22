# **SAM3 Image Segmentation**

**SAM3 Image Segmentation** is a user-friendly web application built with Gradio that leverages the Segment Anything Model 3 (SAM3) from Meta AI to perform zero-shot instance segmentation on images using natural language text prompts. Upload an image, provide a prompt like "cat" or "blue taxi," and the app will automatically detect and highlight matching objects with confidence scores. This app demonstrates SAM3's capabilities for tasks like object detection, semantic segmentation, and interactive editing, all powered by a custom steel-blue themed interface for a sleek user experience.

<img width="1541" height="773" alt="Screenshot 2025-11-22 at 11-59-20 SAM3 Image Segmentation - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/8ae183ac-f21d-44a0-a454-306429ade2fe" />
<img width="1482" height="770" alt="Screenshot 2025-11-22 at 11-59-43 SAM3 Image Segmentation - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/9e4e9811-43a0-4c83-a256-61d7b434a525" />

### Key Features
- **Text-Prompted Segmentation**: Segment objects based on descriptive text (e.g., "player in white", "black cat").
- **Confidence Thresholding**: Adjustable slider to filter segmentation results by confidence score.
- **Interactive Examples**: Built-in examples to quickly test the app.
- **GPU Acceleration**: Optimized for CUDA if available; falls back to CPU.
- **Gradio Interface**: Easy-to-use web UI with annotations overlayed on the original image.
- **Custom Theme**: Steel-blue gradient design for a modern look.

## Requirements

To run this app locally, install the following dependencies. Note that SAM3 requires a recent version of Transformers from a specific commit for compatibility.

```bash
pip install gradio numpy torch peft spaces torchvision pillow opencv-python imageio[pyav] accelerate sentencepiece
pip install git+https://github.com/huggingface/transformers.git@1fba72361e8e0e865d569f7cd15e5aa50b41ac9a
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX series) for faster inference. The app auto-detects and uses CUDA if available.
- **RAM**: At least 8GB (16GB+ recommended for large images).
- **Storage**: ~5GB for model weights (downloaded on first run).

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/SAM3-Image-Segmentation.git
   cd SAM3-Image-Segmentation
   ```

2. **Install Dependencies**:
   Run the pip commands from the [Requirements](#requirements) section above.

3. **Prepare Examples** (Optional):
   Ensure the `examples/` folder contains sample images like `player.jpg`, `goldencat.webp`, and `taxi.jpg`. Download them if needed from public sources or use your own.

4. **Run Locally**:
   ```bash
   python app.py
   ```
   - The app will launch a local web server (typically at `http://127.0.0.1:7860`).
   - Open the URL in your browser to interact with the interface.

### Deployment on Hugging Face Spaces
- This app is optimized for HF Spaces with `@spaces.GPU` decorator.
- Fork the repo on GitHub, create a new Space on [Hugging Face](https://huggingface.co/spaces), and link it to your repo.
- The model will auto-download on first run (ensure your Space has GPU quota).

## Usage

1. **Upload an Image**: Drag-and-drop or select an image file (supports JPG, PNG, WEBP, etc.).
2. **Enter a Text Prompt**: Describe the object to segment (e.g., "cat", "face", "car wheel").
3. **Adjust Threshold**: Use the slider (default: 0.4) to set the minimum confidence for detections.
4. **Click Segment**: The app processes the image and overlays colored masks on detected instances, labeled with scores.
5. **Explore Examples**: Use the built-in examples for quick demos.

### Input/Output Format
- **Input**: PIL Image + Text Prompt + Float Threshold.
- **Output**: Annotated image with segmentation masks (boolean arrays) and labels.

### Example Prompts
- Sports: "player in white" on a soccer image.
- Animals: "black cat" on a pet photo.
- Vehicles: "blue taxi" on a street scene.

## Code Structure

- **`app.py`**: Main Gradio app with model loading, segmentation function, and UI.
- **`examples/`**: Sample images for testing.
- **Custom Theme**: `SteelBlueTheme` class for stylized gradients and shadows.

### Key Functions
- `segment_image(input_image, text_prompt, threshold)`: Core inference using SAM3Processor and Sam3Model.
- Model: Loaded from `"facebook/sam3"` on Hugging Face Hub.

## Troubleshooting

- **Model Loading Error**: Ensure Transformers is installed from the specified commit. Check internet access for downloading weights.
- **CUDA Out of Memory**: Reduce image size or use a lower threshold. Run on CPU by setting `device="cpu"`.
- **No Segments Detected**: Try a simpler prompt or lower the threshold. SAM3 works best on clear, high-contrast objects.
- **Gradio Launch Issues**: Update Gradio (`pip install --upgrade gradio`) or check for port conflicts.

## Contributing

Contributions are welcome! Fork the repo, make changes, and submit a pull request.

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/PRITHIVSAKTHIUR/SAM3-Image-Segmentation.git)

## Acknowledgments

- [Meta AI SAM3](https://huggingface.co/facebook/sam3): The backbone model.
- [Gradio](https://gradio.app/): For the intuitive UI framework.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers): For model handling.
