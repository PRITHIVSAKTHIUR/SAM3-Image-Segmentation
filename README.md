# **SAM3: Segment Anything Model 3**

SAM3: Segment Anything Model 3 is an experimental, multi-functional computer vision application that provides a unified interface for text-conditioned and click-interactive segmentation. Built around Meta's advanced `facebook/sam3` multi-model architecture, this application seamlessly features separate pipelines optimized for text-to-image segmentation, multi-frame video propagation, and high-speed, point-cumulative image tracking. The suite leverages specialized model variants (`Sam3Model`, `Sam3VideoModel`, and `Sam3TrackerModel`) to support highly precise masks over custom objects, bounding boxes, or temporal frame shifts. Fully GPU-accelerated and wrapped in a web workspace with a stylized Citrus theme, SAM3 serves as an advanced sandbox for researchers and developers deploying production-grade, state-of-the-art pixel intelligence workflows.

| Demo 1 | Demo 2 | Demo 3 |
|---|---|---|
| <img src="https://github.com/user-attachments/assets/5cdfe8da-41ff-46d1-af8c-9ceb1c932bf3" width="100%"> | <img src="https://github.com/user-attachments/assets/7412da70-6c12-4b70-a168-5380fb13b5bf" width="100%"> | <img src="https://github.com/user-attachments/assets/3ba0b2d4-30b1-47c2-bb54-d48c24c8e9eb" width="100%"> |

### **Key Features**

* **Text-Conditional Image Segmentation:** Instantly isolates target objects inside static images by matching custom queries (e.g., *"cat"*, *"face"*, *"car wheel"*) against SAM3 instance maps, complete with custom threshold masking.
* **Temporal Video Object Propagation:** Tracks and cuts targeted entities dynamically across successive video frames. By incorporating bfloat16 numerical computation, the video pipeline scales smoothly inside active VRAM constraints.
* **Interactive Point Tracking:** Allows users to interactively click directly on a live input canvas. The model processes each cursor selection as a positive foreground anchor and updates the colored mask overlays instantly.
* **Bespolke Gradio Workspace:** Features an elegant, three-tab layout powered by the Gradio theme engine, integrating clean file-drop inputs, real-time tracking logs, and automated example sets.
* **Supervision and Matplotlib Styling:** Implements structural contour draws and randomized colormap overlays over generated masks to ensure clear human validation on complex textures.

### **Repository Structure**

```text
├── examples/
│   ├── goldencat.webp
│   ├── player.jpg
│   ├── sample_video.mp4
│   ├── sample_video2.mp4
│   └── taxi.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
└── requirements.txt

```

---

### **Installation and Requirements**

To run SAM3 locally, configure a Python environment with the following dependencies. A compatible CUDA-enabled GPU is strongly recommended to handle real-time segmentation and video processing loops.

**Standard PIP Installation**

1. Update pip:

```bash
pip install pip>=26.1.1

```

2. Install dependencies:

```bash
pip install -r requirements.txt

```

#### **Running with `uv` (Recommended)**

`uv` is an extremely fast Python package and project manager written in Rust, ensuring rapid environment setup and complete reproducibility.

**Step 1 — Install `uv**`

* **macOS / Linux:** `curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh`
* **Windows:** `powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/SAM3-Demo.git
cd SAM3-Demo

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

---

### **Core Requirements List**

The application depends on the following primary packages (defined in `requirements.txt`):

```text
transformers==5.9.0
sentencepiece
opencv-python
imageio[pyav]
torchvision
matplotlib
accelerate
kernels
pillow
gradio==6.6.0
spaces
numpy
torch==2.11.0
peft

```

### **Usage**

Once the application is running, open your browser to the local address provided in your terminal (typically `[http://127.0.0.1:7860/](http://127.0.0.1:7860/)`).

1. **Image Segmentation Tab:** Drop an image, enter a target prompt (e.g., *"player in white"*), and click **Segment Image** to see an annotated map.
2. **Video Segmentation Tab:** Drop an MP4 clip, configure maximum frames/timeout conditions, write a tracking prompt, and click **Segment Video** to run mask propagation.
3. **Image Click Segmentation Tab:** Upload your image, and click any element on the preview canvas to see positive foreground anchors auto-generate segmentation layers instantly.

### **License and Source**

* **License:** [SAM License](https://github.com/PRITHIVSAKTHIUR/SAM3-Demo/blob/main/LICENSE)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/SAM3-Demo.git](https://github.com/PRITHIVSAKTHIUR/SAM3-Demo.git)
