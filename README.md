# **[Flux.2 Klein — Small Decoder VAE](https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-Small-Decoder)**

Flux.2 Klein — Small Decoder VAE is an experimental, high-performance image generation and editing application designed to leverage the powerful `black-forest-labs/FLUX.2-klein-4B` distilled model paired strictly with the `FLUX.2-small-decoder`. This application is engineered to test the efficiency and output characteristics of the small decoder architecture via a robust, Citrus-themed Gradio web interface. Operating entirely on CUDA-enabled GPUs with model CPU offloading, the suite provides a seamless workflow for pure text-to-image synthesis, as well as complex image-to-image editing, relighting, and texture enhancement across batch image uploads.

> hf.co/spaces — [prithivmlmods/flux.2-klein-small-decoder](https://huggingface.co/spaces/prithivMLmods/Flux.2-Klein-Small-Decoder)


| example 1 | example 2 |
|-----------|-----------|
| <img src="https://github.com/user-attachments/assets/e77919a0-6568-4d39-a230-6c8cac9216e0" width="100%"> | <img src="https://github.com/user-attachments/assets/08a1bec7-bbbd-4fa2-b718-5593bbc88628" width="100%"> |

### **Key Features**

* **Small Decoder Integration:** Explicitly loads and utilizes the lightweight `black-forest-labs/FLUX.2-small-decoder` Variational Autoencoder alongside the 4B base model, optimizing the decoding pipeline for specific performance testing.
* **Flexible Input Methods:** Supports standard text-to-image generation alongside multi-image Gallery uploads. It intelligently calculates and snaps target resolutions based on the uploaded reference media's aspect ratio.
* **Granular Inference Controls:** Provides a collapsible 'Advanced Settings' panel to manually configure the Generation Seed, Inference Steps, Base Dimensions, and Guidance Scale.
* **Dynamic Resolution Scaling:** Automatically resizes and scales uploaded reference images, maintaining correct proportions while ensuring the dimensions snap to multiples of 8 (bounded by a maximum dimension of 1024x1024) to prevent tensor shape mismatches.

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

To run Flux.2 Klein — Small Decoder VAE locally, you must configure a Python environment equipped to handle advanced compilation and heavy model weights. A modern CUDA-enabled GPU is required.

This repository specifically relies on **PyTorch 2.11.0 and CUDA 13.0** (`--extra-index-url https://download.pytorch.org/whl/cu130`).

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, ensuring rapid virtual environment synchronization and reproducible execution.

**Step 1 — Install `uv**`

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git
cd Flux.2-Klein-Small-Decoder-Only

```

**Step 3 — Initialize the project and install dependencies**
This will automatically parse the `uv.lock` and `requirements.txt` to fetch the correct PyTorch 2.11.0 + cu130 wheels.

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
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, diffusion utilities, and ecosystem structures. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
--extra-index-url https://download.pytorch.org/whl/cu130

git+https://github.com/huggingface/transformers.git@v4.57.6
huggingface-hub
gradio==6.16.0
torch==2.11.0
opencv-python
sentencepiece
torchvision
torchaudio
accelerate
omegaconf
termcolor
diffusers
kernels
imageio
hf_xet
spaces
pyyaml
pillow
numpy
peft
ftfy
av

```

### **Usage**

After setting up your environment and ensuring your dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py

```

The script will initialize the FLUX.2 pipeline and the small decoder VAE, loading them into memory with CPU offloading enabled to optimize VRAM. Once ready, it will expose a local web server (typically at `http://127.0.0.1:7860/`).

Open this address in your browser to access the interface. You can upload reference images into the gallery or leave it empty, input your generation prompt (e.g., *"Change the weather to stormy"*), adjust advanced settings if necessary, and click "Generate Image" to create and view your results.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git)
