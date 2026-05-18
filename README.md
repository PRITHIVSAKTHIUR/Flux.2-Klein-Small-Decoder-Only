# **Flux.2-Klein-Small-Decoder-Only**

Flux.2-Klein-Small-Decoder-Only is an experimental, high-performance image generation and editing application built to exclusively utilize the FLUX.2-klein-4B model paired with the specialized FLUX.2-small-decoder Variational Autoencoder (VAE). Designed to test the efficiency and output characteristics of the small decoder, this suite bypasses traditional UI components in favor of a bespoke, headless Gradio Server implementation. The frontend is engineered with custom HTML, CSS, and JavaScript, presenting a sleek, dark Aubergine-themed workspace. Operating entirely on CUDA-enabled GPUs with model CPU offloading, the application allows users to perform both pure text-to-image synthesis and image-to-image editing with precise control over inference parameters, making it an optimal sandbox for evaluating lightweight diffusion decoding.

### **Key Features**

* **Small Decoder Integration:** Explicitly loads and utilizes the `black-forest-labs/FLUX.2-small-decoder` VAE alongside the distilled 4B base model, optimizing the decoding pipeline for specific performance and aesthetic testing.
* **Custom Headless UI:** Features a completely custom, responsive web interface served via FastAPI. It includes dynamic status logging, a drag-and-drop media zone, and a seamless generation preview stage.
* **Versatile Generation Modes:** Supports pure text-to-image generation as well as image-to-image editing when users upload reference media.
* **Dynamic Resolution Scaling:** Automatically calculates and snaps dimensions to optimal sizes (multiples of 8, bounded to 1024x1024) based on the aspect ratio of any uploaded reference image.
* **Advanced Inference Controls:** Provides a hidden 'Advanced Settings' panel to manually adjust the Generation Seed, Inference Steps, Width, Height, and Guidance Scale.

### **Installation and Requirements**

To run Flux.2-Klein-Small-Decoder-Only locally, configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU to handle the model weights and inference processes.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:

```bash
pip install pip>=26.1

```

**2. Install Core Requirements**
Install the necessary machine learning, diffusion, and web server libraries. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/diffusers.git
transformers==4.57.6
huggingface_hub
sentencepiece
bitsandbytes
torchvision
accelerate
torchao
spaces
hf_xet
gradio
numpy
torch==2.11.0
peft
av

```

### **Usage**

After setting up your environment and ensuring your dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py

```

The script will initialize the FLUX.2 pipeline and the small decoder VAE, loading them into memory with CPU offloading enabled. Once ready, it will expose a local web server (typically at `[http://127.0.0.1:7860/](http://127.0.0.1:7860/)`). Open this address in your browser to access the interface. You can drag and drop reference images, input your generation prompt, adjust advanced settings, and click "Generate" to create and download your results.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git](https://github.com/PRITHIVSAKTHIUR/Flux.2-Klein-Small-Decoder-Only.git)
