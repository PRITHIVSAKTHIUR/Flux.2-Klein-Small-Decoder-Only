# **[Flux.2-Klein-Edit-Ultra-Fast](https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-Edit-Ultra-Fast)**

Flux.2-Klein-Edit-Ultra-Fast (Flux.2-Klein-Small-Decoder-Only) is a high-performance image editing and generation platform powered by the `black-forest-labs/FLUX.2-klein-4B` model paired with the `black-forest-labs/FLUX.2-small-decoder` VAE. The suite enables fast 4-step image-to-image editing, multi-image reference manipulation, and text-to-image generation directly on CUDA hardware.

The application uses a FastAPI backend server (`gradio.Server`) paired with a dark-mode frontend single-page application (SPA). Features include an A/B image comparison slider, a result history filmstrip, quick prompt chips, and a dual-view canvas.

### **Key Features**

* **Optimized Small Decoder VAE:** Uses the `FLUX.2-small-decoder` VAE alongside `FLUX.2-klein-4B` to accelerate decoding cycles while preserving visual details and prompt adherence.
* **Dual-Mode Generation (I2I & T2I):** Performs image editing when input images are supplied, or falls back seamlessly to text-to-image generation if the input gallery is empty.
* **Multi-Image Reference Editing:** Accepts multiple reference inputs to guide transformations—such as transferring accessories or outfits from reference photos onto target subjects.
* **Studio SPA Interface:** An interactive single-page web app built with vanilla web components, offering an A/B image comparison slider, history filmstrip, prompt chips, and drag-and-drop file uploaders.
* **Automatic Aspect-Ratio Snapping:** Calculates dimensions from the first input image, scaling parameters to fit within 1024px while snapping width and height to multiples of 8.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── I1.jpg
│   └── I2.jpg
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the Flux.2-Klein-Edit-Ultra-Fast environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Minimum Python **3.10.13** or above is required; Python **3.12** or **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for best compatibility.
* **CUDA Version:** CUDA **13.0** is recommended (`--extra-index-url https://download.pytorch.org/whl/cu130`), matching the environment used on the live Hugging Face demo.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git
cd Flux.2-Klein-Small-Decoder-Only
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.22.0
av==17.1.0
spaces==0.51.1
huggingface-hub==1.24.0
```

### **Usage**

Once the web server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Asset (Optional):** Drag and drop images into the main canvas workspace, paste an image from your clipboard, or click the upload icon in the left rail. Leave empty for text-to-image generation.
2. **Refine Instructions:** Type your instructions inside the prompt field, or click one of the **Quick Prompts** chips to fill it. Press ⌘/Ctrl + Enter or click **Edit Image**.
3. **Compare & Chain:** Use the **Compare** tool on the left rail to view an A/B slider of the before and after states. Click **Use as Input** to chain multiple edits sequentially.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-Edit-Ultra-Fast](https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-Edit-Ultra-Fast)
