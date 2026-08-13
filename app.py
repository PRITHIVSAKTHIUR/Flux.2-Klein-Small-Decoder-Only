import os
import gc
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
import random
import base64
import json
from io import BytesIO
from PIL import Image
from typing import Tuple

from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

print("Loading Small Decoder VAE...")
vae_small = AutoencoderKLFlux2.from_pretrained(
    "black-forest-labs/FLUX.2-small-decoder",
    torch_dtype=dtype,
).to(device)

print("Loading 4B Distilled model (Small Decoder VAE)...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    vae=vae_small,
    torch_dtype=dtype,
).to(device)
print("Pipeline loaded directly to CUDA.")

# ── Examples Config ───────────────────────────────────────────────────────────
EXAMPLES_CONFIG = [
    {"images": ["examples/I1.jpg", "examples/I2.jpg"], "prompt": "Make her wear these glasses in Image 2."},
    {"images": ["examples/1.jpg"], "prompt": "Change the weather to stormy."},
    {"images": ["examples/2.jpg"], "prompt": "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition."},
    {"images": ["examples/3.jpg"], "prompt": "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent."},
    {"images": ["examples/4.jpg"], "prompt": "Make the texture high-resolution."},
    {"images": [], "prompt": "A futuristic cyberpunk cityscape at night, neon lights reflecting in puddles, flying cars in the background."},
]

def calc_dimensions(pil_img: Image.Image) -> Tuple[int, int]:
    """Calculates dimensions preserving aspect ratio, snapped to multiples of 8."""
    iw, ih = pil_img.size
    aspect = iw / ih

    if aspect >= 1:
        new_width  = 1024
        new_height = int(round(1024 / aspect))
    else:
        new_height = 1024
        new_width  = int(round(1024 * aspect))

    new_width  = max(256, min(1024, round(new_width  / 8) * 8))
    new_height = max(256, min(1024, round(new_height / 8) * 8))
    return new_width, new_height

def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""

def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""

def build_client_config():
    """Static config consumed by the frontend: example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "prompt": ex["prompt"],
        })
    return {
        "examples": examples,
    }

print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples.")

def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images

def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="Flux.2-Klein-Edit-Ultra-Fast")

@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
) -> dict:
    """Edits an image or generates from text with FLUX.2 Klein 4B."""
    gc.collect()
    torch.cuda.empty_cache()

    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a prompt.")

    pil_images = b64_to_pil_list(images_b64_json)
    
    if pil_images:
        # Calculate dims from first image and resize all
        calc_w, calc_h = calc_dimensions(pil_images[0])
        width, height = calc_w, calc_h
        processed_images = [
            img.resize((width, height), LANCZOS).convert("RGB") 
            for img in pil_images
        ]
        image_input = processed_images if len(processed_images) > 1 else processed_images[0]
    else:
        image_input = None

    # Ensure dimensions are multiples of 8
    final_width  = max(256, min(1024, round(int(width)  / 8) * 8))
    final_height = max(256, min(1024, round(int(height) / 8) * 8))

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    kwargs = dict(
        prompt=prompt,
        height=final_height,
        width=final_width,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    if image_input is not None:
        kwargs["image"] = image_input

    try:
        result_image = pipe(**kwargs).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {"images": b64_list, "prompt": ex["prompt"], "names": names, "status": "ok"}


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)