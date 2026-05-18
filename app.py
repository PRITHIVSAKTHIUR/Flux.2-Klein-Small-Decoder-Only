import os
import io
import gc
import uuid
import json
import base64
import random
import zipfile
import threading
from pathlib import Path
from typing import List, Optional

import spaces
import numpy as np
import torch
from PIL import Image

from gradio import Server
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

# --- App Configuration & Directories ---
app = Server()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
EXAMPLES_DIR = BASE_DIR / "examples"

STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

dtype  = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    DEVICE_LABEL = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
else:
    DEVICE_LABEL = str(device).lower()

# --- Model Loading ---
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
pipe.enable_model_cpu_offload()

pipe_lock = threading.Lock()

# --- Utility Functions ---
def calc_dimensions(pil_img: Image.Image):
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

def parse_and_resize_images(image_paths: List[str], width: int, height: int):
    if not image_paths:
        return None

    resized = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            resized.append(img.resize((width, height), Image.LANCZOS))
        except Exception as e:
            print(f"Skipping invalid image: {e}")

    return resized if resized else None

def run_pipeline(kwargs, seed):
    with pipe_lock:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(**kwargs, generator=gen).images[0]
    return result

def save_image(img: Image.Image, prefix: str = "output") -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    path = OUTPUT_DIR / filename
    img.save(path, format="PNG")
    return filename

# --- Inference Function ---
@spaces.GPU(size="xlarge")
def infer(
    prompt: str,
    image_paths: List[str] = None,
    seed: int = 42,
    randomize_seed: bool = False,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not prompt or not prompt.strip():
        raise ValueError("Please enter a prompt.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    image_list = None
    if image_paths and len(image_paths) > 0:
        try:
            first_pil = Image.open(image_paths[0]).convert("RGB")
            width, height = calc_dimensions(first_pil)
            image_list = parse_and_resize_images(image_paths, width, height)
        except Exception as e:
            print(f"Error processing upload: {e}")

    width  = max(256, min(MAX_IMAGE_SIZE, round(int(width)  / 8) * 8))
    height = max(256, min(MAX_IMAGE_SIZE, round(int(height) / 8) * 8))

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    if image_list is not None:
        kwargs["image"] = image_list

    result = run_pipeline(kwargs, seed)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, seed


# --- FastAPI Endpoints ---
def get_example_items():
    return [
        {
            "urls": ["/example-file/I1.jpg", "/example-file/I2.jpg"],
            "prompt": "Make her wear these glasses in Image 2."
        },
        {
            "urls": ["/example-file/1.jpg"],
            "prompt": "Change the weather to stormy."
        },
        {
            "urls": ["/example-file/2.jpg"],
            "prompt": "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition."
        },
        {
            "urls": ["/example-file/3.jpg"],
            "prompt": "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent."
        },
        {
            "urls": ["/example-file/4.jpg"],
            "prompt": "Make the texture high-resolution."
        }
    ]

@app.get("/example-file/{filename}")
async def example_file(filename: str):
    path = EXAMPLES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Example not found"}, status_code=404)
    return FileResponse(path)

@app.get("/download/{filename}")
async def download_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, filename=filename, media_type="image/png")

@app.post("/api/generate")
async def generate_image(
    prompt: str = Form(...),
    seed: str = Form("0"),
    randomize_seed: str = Form("true"),
    width: str = Form("1024"),
    height: str = Form("1024"),
    steps: str = Form("4"),
    guidance: str = Form("1.0"),
    images: Optional[List[UploadFile]] = File(None),
):
    temp_paths = []
    try:
        image_paths = []
        if images:
            for upload in images:
                if not upload.filename: continue
                suffix = Path(upload.filename).suffix or ".png"
                temp_path = OUTPUT_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
                content = await upload.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_paths.append(str(temp_path))
                image_paths.append(str(temp_path))

        result, used_seed = infer(
            prompt=prompt,
            image_paths=image_paths,
            seed=int(seed),
            randomize_seed=(randomize_seed.lower() == "true"),
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        )

        filename = save_image(result, prefix="output")

        return JSONResponse({
            "success": True,
            "seed": used_seed,
            "url": f"/download/{filename}",
            "filename": filename,
            "device": DEVICE_LABEL,
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)

# --- Frontend ---
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    examples = get_example_items()
    examples_json = json.dumps(examples)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Flux.2 Klein — Small Decoder</title>
  <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ub-aubergine: #2C001E;
      --ub-aubergine-dark: #1f0015;
      --ub-orange: #E95420;
      --ub-orange-hover: #c4461a;
      --ub-panel: #3D3D3D;
      --ub-panel-light: #4f4f4f;
      --ub-border: rgba(255,255,255,0.1);
      --ub-text: #FFFFFF;
      --ub-muted: #b0b0b0;
      --ub-input: #2b2b2b;
      --panel-radius: 8px;
    }}

    * {{ box-sizing: border-box; font-family: 'Ubuntu', sans-serif; }}

    body {{
      margin: 0; padding: 0;
      background: var(--ub-aubergine);
      color: var(--ub-text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    .topbar {{
      background: var(--ub-aubergine-dark);
      padding: 16px 24px;
      border-bottom: 1px solid var(--ub-border);
      text-align: center;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--ub-orange);
    }}

    .container {{
      max-width: 1300px;
      margin: 0 auto;
      padding: 30px 20px;
      flex: 1;
      width: 100%;
    }}

    .header-text {{
      text-align: center;
      margin-bottom: 30px;
    }}
    .header-text h1 {{
      margin: 0 0 10px 0;
      font-size: 2.2rem;
    }}
    .header-text p {{
      color: var(--ub-muted);
      margin: 0;
    }}

    /* LAYOUT */
    .layout {{
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 24px;
      align-items: stretch;
      height: 650px;
    }}

    .panel {{
      background: var(--ub-panel);
      border-radius: var(--panel-radius);
      box-shadow: 0 8px 24px rgba(0,0,0,0.2);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      height: 100%;
    }}

    .panel-header {{
      padding: 16px 20px;
      background: rgba(0,0,0,0.2);
      border-bottom: 1px solid var(--ub-border);
      font-weight: 500;
      font-size: 1.1rem;
      flex-shrink: 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .panel-body-scroll {{
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }}

    /* Input Forms */
    .form-group {{ margin-bottom: 20px; flex-shrink: 0; }}
    .label {{
      display: block; font-weight: 500; font-size: 14px;
      color: var(--ub-muted); margin-bottom: 8px;
    }}

    .textarea, .input {{
      width: 100%;
      background: var(--ub-input);
      border: 1px solid var(--ub-border);
      color: var(--ub-text);
      padding: 12px;
      border-radius: 4px;
      outline: none;
      font-size: 14px;
    }}
    .textarea:focus, .input:focus {{ border-color: var(--ub-orange); }}
    .textarea {{ min-height: 100px; resize: vertical; }}

    /* Upload Zone */
    .upload-zone {{
      background: var(--ub-input);
      border: 1px dashed var(--ub-muted);
      border-radius: 4px;
      padding: 15px;
      text-align: center;
      cursor: pointer;
      transition: background 0.2s, border-color 0.2s;
      min-height: 100px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }}
    .upload-zone:hover, .upload-zone.dragover {{
      border-color: var(--ub-orange);
      background: rgba(233,84,32,0.05);
    }}
    .upload-zone input[type="file"] {{ display: none; }}
    .upload-text {{ pointer-events: none; color: var(--ub-muted); }}

    .preview-grid {{
      display: none;
      grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
      gap: 10px;
      width: 100%;
    }}
    .thumb {{
      position: relative; aspect-ratio: 1;
      border-radius: 4px; overflow: hidden;
      border: 1px solid var(--ub-border);
    }}
    .thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .thumb-remove {{
      position: absolute; top: 4px; right: 4px;
      background: rgba(0,0,0,0.7); color: white;
      border: none; border-radius: 50%; width: 20px; height: 20px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; font-size: 12px;
    }}

    .add-more-btn {{
      display: flex; align-items: center; justify-content: center;
      border: 2px dashed var(--ub-muted); border-radius: 4px;
      color: var(--ub-muted); font-size: 26px; cursor: pointer;
      aspect-ratio: 1; transition: all 0.2s; background: transparent;
    }}
    .add-more-btn:hover {{
      border-color: var(--ub-orange); color: var(--ub-orange);
      background: rgba(233,84,32,0.05);
    }}

    /* Advanced Accordion */
    .advanced-toggle {{
      width: 100%; background: none; border: none; color: var(--ub-orange);
      text-align: left; padding: 10px 0; font-weight: 500; cursor: pointer;
      display: flex; justify-content: space-between; align-items: center;
      flex-shrink: 0;
    }}
    .advanced-icon {{ font-weight: bold; font-size: 18px; line-height: 1; }}
    .advanced-body {{ display: none; padding-top: 10px; flex-shrink: 0; }}
    .advanced-body.open {{ display: block; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}

    /* Status Container */
    .status-container {{
      margin-top: 20px; margin-bottom: 20px;
      border: 1px solid var(--ub-border); border-radius: 4px;
      background: #200014; display: flex; flex-direction: column;
      flex: 1; min-height: 100px; max-height: 200px;
    }}
    .status-header {{
      padding: 8px 12px; font-size: 11px; font-weight: 700;
      color: var(--ub-muted); background: rgba(0,0,0,0.4);
      border-bottom: 1px solid var(--ub-border); text-transform: uppercase;
      letter-spacing: 0.5px; flex-shrink: 0;
    }}
    .status-log {{
      padding: 10px; font-family: 'Courier New', Courier, monospace;
      font-size: 12px; color: #eeeeee; overflow-y: auto;
      flex: 1; display: flex; flex-direction: column; gap: 4px;
    }}
    .log-time {{ color: #777; margin-right: 8px; }}
    .log-info {{ color: #5bc0eb; }}
    .log-success {{ color: #9bc53d; }}
    .log-error {{ color: #ff5e5b; }}

    /* Buttons */
    .btn {{
      width: 100%; padding: 14px; border: none; border-radius: 4px;
      font-size: 16px; font-weight: 700; cursor: pointer;
      transition: opacity 0.2s, background 0.2s; flex-shrink: 0;
    }}
    .btn-primary {{
      background: var(--ub-orange); color: white;
      box-shadow: 0 4px 12px rgba(233,84,32,0.3);
    }}
    .btn-primary:hover {{ background: var(--ub-orange-hover); }}
    .btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}

    /* Top-Right Download Icon */
    .action-icon {{
      display: none; background: none; border: none; color: var(--ub-muted);
      cursor: pointer; padding: 4px; transition: color 0.2s;
    }}
    .action-icon:hover {{ color: var(--ub-orange); }}

    /* OUTPUT PANEL */
    .panel-body-output {{
      flex: 1; display: flex; flex-direction: column;
      padding: 0; position: relative;
    }}
    .output-stage {{
      position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      background: #111; overflow: hidden; display: flex;
      align-items: center; justify-content: center;
    }}
    .output-empty {{ color: var(--ub-muted); text-align: center; z-index: 1; }}

    .output-img {{
      position: absolute; top: 0; left: 0; width: 100%; height: 100%;
      object-fit: contain; display: none; user-select: none;
    }}

    /* LOADER */
    .loader {{
      position: absolute; inset: 0;
      background: rgba(20, 0, 10, 0.7);
      backdrop-filter: blur(6px);
      display: none; flex-direction: column;
      align-items: center; justify-content: center; z-index: 20;
    }}
    .spinner-single {{
      width: 55px; height: 55px;
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-top-color: var(--ub-orange);
      border-radius: 50%;
      animation: spin 1s cubic-bezier(0.4, 0.0, 0.2, 1) infinite;
      margin-bottom: 20px;
    }}
    .loader-text {{
      font-weight: 500;
      font-size: 15px;
      color: #ffffff;
      letter-spacing: 1px;
      animation: pulse 1.5s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.5; }}
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}

    /* Seed Badge */
    .seed-badge {{
      display: none;
      position: absolute; bottom: 12px; right: 12px;
      background: rgba(0,0,0,0.6); color: #ccc; padding: 5px 10px;
      border-radius: 20px; font-size: 12px; backdrop-filter: blur(4px);
      z-index: 5;
    }}

    /* Examples */
    .examples-section {{ margin-top: 40px; }}
    .examples-section h3 {{ border-bottom: 1px solid var(--ub-border); padding-bottom: 10px; }}
    .examples-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px;
    }}
    .ex-card {{
      background: var(--ub-panel); border-radius: 4px; overflow: hidden;
      cursor: pointer; transition: transform 0.2s, box-shadow 0.2s;
    }}
    .ex-card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }}
    .ex-card-img-wrap {{ width: 100%; aspect-ratio: 1; display: flex; background: #000; }}
    .ex-card-img-wrap img {{ height: 100%; object-fit: cover; }}
    .ex-card p {{ padding: 12px; margin: 0; font-size: 13px; color: var(--ub-muted); line-height: 1.4; }}

    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; height: auto; }}
      .panel-body-output {{ height: 450px; flex: none; }}
      .output-stage {{ position: relative; height: 100%; }}
    }}
  </style>
</head>
<body>

  <div class="topbar">Flux.2 Klein 4B — Small Decoder</div>

  <div class="container">
    <div class="header-text">
      <h1>Flux.2 Klein — Small Decoder VAE</h1>
      <p>Upload an image (optional) and enter a prompt to generate with the 4B distilled model.</p>
    </div>

    <div class="layout">
      <!-- Left: Settings Panel -->
      <div class="panel">
        <div class="panel-header">Settings</div>
        <div class="panel-body-scroll">
          <div class="form-group">
            <label class="label">Input Images (Optional)</label>
            <div class="upload-zone" id="dropZone">
              <input type="file" id="fileInput" multiple accept="image/*" />
              <div class="upload-text" id="uploadText">Click or Drag & Drop images here</div>
              <div class="preview-grid" id="previewGrid"></div>
            </div>
          </div>

          <div class="form-group">
            <label class="label">Prompt</label>
            <textarea id="promptInput" class="textarea" placeholder="Describe the edit or generation..."></textarea>
          </div>

          <button class="advanced-toggle" id="advToggle">
            <span>Advanced Settings</span> <span class="advanced-icon" id="advIcon">+</span>
          </button>

          <div class="advanced-body" id="advBody">
            <div class="grid-2">
              <div class="form-group">
                <label class="label">Seed</label>
                <input type="number" id="seed" class="input" value="0">
              </div>
              <div class="form-group">
                <label class="label">Steps</label>
                <input type="number" id="steps" class="input" value="4">
              </div>
              <div class="form-group">
                <label class="label">Width</label>
                <input type="number" id="width" class="input" value="1024" step="8">
              </div>
              <div class="form-group">
                <label class="label">Height</label>
                <input type="number" id="height" class="input" value="1024" step="8">
              </div>
              <div class="form-group" style="grid-column: span 2;">
                <label class="label">Guidance Scale</label>
                <input type="number" id="guidance" class="input" value="1.0" step="0.1">
              </div>
              <div class="form-group" style="grid-column: span 2;">
                <label style="display:flex; align-items:center; gap:8px; font-size:14px; color:var(--ub-text);">
                  <input type="checkbox" id="randomize" checked> Randomize Seed
                </label>
              </div>
            </div>
          </div>

          <div class="status-container">
            <div class="status-header">Execution Log</div>
            <div class="status-log" id="statusLog">
              <div><span class="log-time">[{DEVICE_LABEL}]</span><span>System Ready...</span></div>
            </div>
          </div>

          <button class="btn btn-primary" id="runBtn">Generate</button>
        </div>
      </div>

      <!-- Right: Output Panel -->
      <div class="panel">
        <div class="panel-header">
          <span>Output</span>
          <button id="downloadBtn" class="action-icon" title="Download Image">
            <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
          </button>
        </div>
        <div class="panel-body-output">
          <div class="output-stage" id="outputStage">
            <div class="output-empty" id="outputEmpty">
              <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" style="margin-bottom:10px; opacity:0.5;">
                <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              <div>Result will appear here</div>
            </div>

            <img id="outputImg" class="output-img" alt="Generated Output" />

            <div class="seed-badge" id="seedBadge"></div>

            <div class="loader" id="loader">
              <div class="spinner-single"></div>
              <div class="loader-text">Generating image...</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Examples Section -->
    <div class="examples-section">
      <h3>Examples</h3>
      <div class="examples-grid" id="examplesGrid"></div>
    </div>
  </div>

  <script>
    const examples = {examples_json};
    let filesState = [];
    let currentFilename = "";

    const dropZone      = document.getElementById('dropZone');
    const fileInput     = document.getElementById('fileInput');
    const previewGrid   = document.getElementById('previewGrid');
    const uploadText    = document.getElementById('uploadText');
    const promptInput   = document.getElementById('promptInput');
    const runBtn        = document.getElementById('runBtn');
    const downloadBtn   = document.getElementById('downloadBtn');
    const statusLog     = document.getElementById('statusLog');
    const outputStage   = document.getElementById('outputStage');
    const outputImg     = document.getElementById('outputImg');
    const outputEmpty   = document.getElementById('outputEmpty');
    const loader        = document.getElementById('loader');
    const seedBadge     = document.getElementById('seedBadge');

    function logMsg(msg, styleClass="") {{
      const div = document.createElement('div');
      const timeStr = new Date().toLocaleTimeString('en-US', {{hour12:false}});
      div.innerHTML = `<span class="log-time">[${{timeStr}}]</span><span class="${{styleClass}}">${{msg}}</span>`;
      statusLog.appendChild(div);
      statusLog.scrollTop = statusLog.scrollHeight;
    }}

    // --- Advanced Toggle ---
    document.getElementById('advToggle').onclick = function() {{
      const body = document.getElementById('advBody');
      body.classList.toggle('open');
      document.getElementById('advIcon').innerText = body.classList.contains('open') ? '−' : '+';
    }};

    // --- File Upload Logic ---
    function renderPreviews() {{
      previewGrid.innerHTML = '';
      if (filesState.length > 0) {{
        uploadText.style.display = 'none';
        previewGrid.style.display = 'grid';

        filesState.forEach((f, i) => {{
          const div = document.createElement('div');
          div.className = 'thumb';
          const img = document.createElement('img');
          img.src = URL.createObjectURL(f);
          const btn = document.createElement('button');
          btn.className = 'thumb-remove';
          btn.innerText = '×';
          btn.onclick = (e) => {{ e.stopPropagation(); filesState.splice(i, 1); renderPreviews(); }};
          div.appendChild(img); div.appendChild(btn);
          previewGrid.appendChild(div);
        }});

        const addBtn = document.createElement('div');
        addBtn.className = 'add-more-btn';
        addBtn.innerHTML = '+';
        addBtn.onclick = (e) => {{ e.stopPropagation(); fileInput.click(); }};
        previewGrid.appendChild(addBtn);
      }} else {{
        uploadText.style.display = 'block';
        previewGrid.style.display = 'none';
      }}
    }}

    dropZone.onclick = (e) => {{ if (e.target === dropZone || e.target === uploadText) fileInput.click(); }};
    fileInput.onchange = (e) => {{ filesState.push(...Array.from(e.target.files)); renderPreviews(); fileInput.value=''; }};
    dropZone.ondragover = (e) => {{ e.preventDefault(); dropZone.classList.add('dragover'); }};
    dropZone.ondragleave = () => dropZone.classList.remove('dragover');
    dropZone.ondrop = (e) => {{
      e.preventDefault(); dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {{ filesState.push(...Array.from(e.dataTransfer.files)); renderPreviews(); }}
    }};

    // --- Examples Logic ---
    async function loadExample(urls, text) {{
      filesState = [];
      renderPreviews();
      promptInput.value = text;
      logMsg("Loading example: " + text, "log-info");

      try {{
        for (let i = 0; i < urls.length; i++) {{
          const res = await fetch(urls[i]);
          const blob = await res.blob();
          const filename = urls[i].split('/').pop();
          filesState.push(new File([blob], filename, {{type: blob.type}}));
        }}
        renderPreviews();
        window.scrollTo({{top: 0, behavior: 'smooth'}});
        setTimeout(() => {{
          logMsg("Example loaded. Starting generation...", "log-info");
          runBtn.click();
        }}, 500);
      }} catch (e) {{
        logMsg("Failed to load example images.", "log-error");
        alert('Failed to load example image.');
      }}
    }}

    const exGrid = document.getElementById('examplesGrid');
    examples.forEach(ex => {{
      const card = document.createElement('div');
      card.className = 'ex-card';

      let imgHTML = '';
      if (ex.urls.length > 1) {{
        imgHTML = `
          <div class="ex-card-img-wrap">
            <img src="${{ex.urls[0]}}" style="width:50%; border-right:1px solid #000;">
            <img src="${{ex.urls[1]}}" style="width:50%;">
          </div>`;
      }} else {{
        imgHTML = `<div class="ex-card-img-wrap"><img src="${{ex.urls[0]}}" style="width:100%;"></div>`;
      }}

      card.innerHTML = `${{imgHTML}}<p>${{ex.prompt}}</p>`;
      card.onclick = () => loadExample(ex.urls, ex.prompt);
      exGrid.appendChild(card);
    }});

    // --- Download Logic ---
    downloadBtn.onclick = () => {{
      if (!currentFilename) return;
      logMsg("Downloading image...", "log-info");
      window.location.href = `/download/${{currentFilename}}`;
    }};

    // --- Form Submission ---
    runBtn.onclick = async () => {{
      const prompt = promptInput.value.trim();
      if (!prompt) {{
        logMsg("Validation failed: Prompt is empty.", "log-error");
        return alert("Enter a prompt");
      }}

      logMsg("Initializing generation...", "log-info");

      const fd = new FormData();
      fd.append('prompt', prompt);
      fd.append('seed', document.getElementById('seed').value);
      fd.append('randomize_seed', document.getElementById('randomize').checked);
      fd.append('width', document.getElementById('width').value);
      fd.append('height', document.getElementById('height').value);
      fd.append('steps', document.getElementById('steps').value);
      fd.append('guidance', document.getElementById('guidance').value);

      filesState.forEach(f => fd.append('images', f));

      loader.style.display = 'flex';
      runBtn.disabled = true;
      downloadBtn.style.display = 'none';
      seedBadge.style.display = 'none';

      logMsg("Sending request... Running Small Decoder VAE model.", "log-info");

      try {{
        const res = await fetch('/api/generate', {{ method: 'POST', body: fd }});
        const data = await res.json();

        if (data.success) {{
          logMsg(`Success! Used seed: ${{data.seed}}`, "log-success");

          currentFilename = data.filename;
          outputImg.src = data.url;

          outputImg.onload = () => {{
            outputEmpty.style.display = 'none';
            outputImg.style.display = 'block';
            downloadBtn.style.display = 'block';
            seedBadge.innerText = `Seed: ${{data.seed}}`;
            seedBadge.style.display = 'block';
          }};
        }} else {{
          logMsg("Error: " + data.error, "log-error");
          alert('Error: ' + data.error);
        }}
      }} catch (e) {{
        logMsg("Network or server connection failed.", "log-error");
        alert('Failed to connect to server.');
      }} finally {{
        loader.style.display = 'none';
        runBtn.disabled = false;
        logMsg("Ready for next input.", "");
      }}
    }};
  </script>
</body>
</html>
"""

app.launch()
