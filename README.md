# **Flux.2 Klein — Small Decoder VAE**

Flux.2 Klein — Small Decoder VAE is an advanced, experimental image generation and editing application built entirely on the `black-forest-labs/FLUX.2-klein-4B` base model, specifically paired with the newly introduced `FLUX.2-small-decoder` Variational Autoencoder. Designed to optimize visual inference and test alternative latent decoding, this suite provides a user-friendly Gradio interface for both Text-to-Image synthesis and structural Image-to-Image transformations. Featuring CPU offloading, automatic aspect ratio preservation, and dynamic dimension snapping, this application serves as an ideal workspace for iterating on highly detailed, small-decoder-driven diffusion imagery.

| example 1 | example 2 |
|-----------|-----------|
| <img src="https://github.com/user-attachments/assets/e77919a0-6568-4d39-a230-6c8cac9216e0" width="100%"> | <img src="https://github.com/user-attachments/assets/08a1bec7-bbbd-4fa2-b718-5593bbc88628" width="100%"> |

### **Key Features**

* **Small Decoder Integration:** Directly integrates the `black-forest-labs/FLUX.2-small-decoder`, allowing users to test and generate imagery specifically tuned through this highly efficient autoencoder variant.
* **Dual-Mode Inference:** Supports pure Text-to-Image generation alongside multi-image Image-to-Image editing (e.g., style transfer, weather alteration, relighting) using Gradio's Gallery inputs.
* **Dynamic Dimension Scaling:** Automatically calculates and adapts generation resolutions based on uploaded reference images, preserving original aspect ratios while securely snapping dimensions to multiples of 8 (up to $1024 \times 1024$).
* **Advanced Generation Controls:** Includes a collapsible settings panel to manually override and fine-tune Width, Height, Inference Steps, Guidance Scale, and Seed configurations.
* **GPU Memory Optimization:** Implements `enable_model_cpu_offload()` and aggressive garbage collection to ensure the 4B parameter model runs efficiently on consumer-grade hardware.

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
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To configure this application locally, set up a Python environment with the following dependencies. A dedicated CUDA-capable GPU is required to load and execute the models.

**⚠️ Critical System Requirement:** This application requires **PyTorch 2.11.0 and CUDA 13.0**. The dependencies explicitly pull from the matching wheel index: `https://download.pytorch.org/whl/cu130`.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, which guarantees rapid virtual environment synchronization and reproducible execution.

**Step 1 — Install `uv**`

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git
cd Flux.2-Klein-Small-Decoder-Only

```

**Step 3 — Initialize the project and install dependencies**
This will read the requirements and fetch the required PyTorch 2.11.0 + cu130 wheels automatically.

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

#### **Standard PIP Installation**

**1. Install Pre-requirements**
Ensure your local system package manager is upgraded:

```bash
pip install pip>=26.1

```

**2. Install Core Dependencies**
Install the primary deep learning stack, diffusion utilities, and ecosystem structures. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
--extra-index-url https://download.pytorch.org/whl/cu130

git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/diffusers.git
git+https://github.com/huggingface/peft.git
huggingface-hub
gradio==6.16.0
torch==2.11.0
opencv-python
sentencepiece
torchvision
torchaudio
omegaconf
termcolor
kernels
imageio
hf_xet
spaces
pyyaml
pillow
numpy
ftfy
av

```

### **Usage**

Once the Gradio application initializes, load the dashboard by pointing your browser to the local loopback endpoint (typically `http://127.0.0.1:7860/`).

1. **Upload Reference (Optional):** Drop one or more images into the Input Images gallery to trigger Image-to-Image mode. The app will automatically adapt the generation dimensions to match the first image's aspect ratio.
2. **Write Instruction:** Type a descriptive instruction or prompt in the text box (e.g., *"Change the weather to stormy"* or *"A futuristic cyberpunk cityscape at night"*).
3. **Advanced Settings:** Expand the Advanced Settings accordion to tweak Inference Steps (default is 4 for distilled execution), Guidance Scale, or manually override the target width and height.
4. **Generate:** Click **Generate Image**. The backend will process the generation and display the resulting output alongside the specific execution seed used.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git)
