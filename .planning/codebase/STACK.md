# Tech Stack

**Analysis Date:** 2026-04-07

## Languages & Runtime

- **Primary:** Python 3.12 (`.python-version`), requires >=3.11 (`pyproject.toml`)
- **Runtime:** CPython on Windows (win32), CUDA 12.8 enabled

## Package Manager

- **uv** — lockfile present (`uv.lock`), used for dependency resolution
- PyT wheels sourced from `https://download.pytorch.org/whl/cu128` (explicit index)

## Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| **PyTorch** | >=2.9.1 | Core ML computation, GPU tensor ops |
| **Diffusers** | >=0.36.0 | Diffusion model pipelines (SVD-based) |
| **Transformers** | >=5.1.0 | CLIP image encoder for SVD |
| **PySide6** | >=6.11.0 | Qt-based GUI (`splatting_gui_qt.py`) |
| **Tkinter + ttkthemes** | >=3.3.0 | Legacy GUIs (`splatting_gui.py`, `depthcrafter_gui_seg.py`, `inpainting_gui.py`, `merging_gui.py`) |
| **TkinterDnD2** | >=0.4.3 | Drag-and-drop support in Tkinter GUIs |
| **Accelerate** | >=1.12.0 | Model offloading and device management |
| **xFormers** | >=0.0.33 | Memory-efficient attention |
| **Triton** | >=3.6.0 | GPU kernel optimization (Windows) |

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.9.1 | Deep learning framework, GPU compute |
| `torchvision` | >=0.24.1 | Image/video transforms |
| `torchaudio` | >=2.9.1 | Audio support (bundled with PyTorch) |
| `diffusers` | >=0.36.0 | Stable Video Diffusion pipeline, inpainting |
| `transformers` | >=5.1.0 | CLIP-ViT image encoder |
| `numpy` | <2.0.0 | Array operations (pinned <2 for compatibility) |
| `opencv-python` | >=4.11.0.86 | Image processing, video I/O fallback |
| `decord` | >=0.6.0 | Fast video reading |
| `imageio` | >=2.37.2 | Image/video read/write |
| `imageio-ffmpeg` | >=0.6.0 | FFmpeg integration for imageio |
| `moviepy` | >=2.1.1 | Video clip manipulation |
| `mediapy` | >=1.2.6 | Simple video save with FFmpeg args |
| `moderngl` | >=5.12.0 | OpenGL rendering (point cloud visualization) |
| `openexr` | >=3.4.4 | EXR file support for high-bit-depth depth maps |
| `imath` | >=0.0.2 | OpenEXR math utilities |
| `matplotlib` | >=3.10.8 | Plotting/visualization |
| `safetensors` | latest | Safe model weight loading |
| `huggingface-hub` | >=1.7.1 | Model download from HuggingFace |
| `pyperclip` | >=1.11.0 | Clipboard operations |
| `fire` | >=0.7.1 | CLI interface generation |

### Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ruff` | >=0.15.0 | Linting and formatting |

## Configuration

- **`pyproject.toml`** — Project metadata, dependencies, uv source config, ruff settings
- **`uv.lock`** — Frozen dependency resolution (2213 lines)
- **`.python-version`** — Python version pin: `3.12`
- **`config_depthcrafter.json`** — DepthCrafter GUI settings (paths, inference params, merge options)
- **`config_inpaint.json`** — Inpainting GUI settings (paths, chunk sizes, encoding)
- **`config_merging.mergecfg`** — Merging GUI configuration
- **`config_splat.splatcfg`** — Splatting GUI configuration
- **`dependency/*.json`** — Help text JSON files (`inpaint_help.json`, `merge_help.json`, `splatter_help.json`, `help_content.json`)

### Ruff Settings (`pyproject.toml`)

- Line length: 120
- Indent: 4 spaces
- Selects: E, F, W
- Ignores: E501 (line length), F841 (unused variable — unfixable)

## Build/Dev Tools

- **uv** — Fast Python package installer/resolver
- **ruff** — Linter/formatter (configured in `pyproject.toml`)
- **FFmpeg** — External dependency (system-level), required for video encoding/decoding
- **CUDA Toolkit 12.8** — External dependency (system-level), required for GPU acceleration

## Platform Requirements

- **OS:** Windows (primary), with cross-platform markers in lockfile for Linux/macOS
- **GPU:** NVIDIA with CUDA 12.8 support (xFormers, Triton, NVENC encoding)
- **System tools:** FFmpeg on PATH, Git on PATH

## Model Weights (Local)

Located in `weights/`:
- `weights/stable-video-diffusion-img2vid-xt-1-1/` — SVD image-to-video model (image encoder + VAE)
- `weights/DepthCrafter/` — DepthCrafter UNet for video depth estimation
- `weights/StereoCrafter/` — StereoCrafter inpainting UNet
- `dependency/iw3_sod_v1_20260122.pth` — U2Net salient object detection model

## Notable Patterns

- **Dual GUI frameworks:** Tkinter (legacy, 4 GUIs) and PySide6 (new, `splatting_gui_qt.py`) coexist
- **Sidecar JSON configs:** Per-video `.json` sidecar files store splatting parameters (`core/common/sidecar_manager.py`)
- **Custom diffusion pipelines:** `pipelines/stereo_video_inpainting.py` extends `DiffusersPipeline` for stereo inpainting
- **DepthCrafter custom pipeline:** `depthcrafter/depth_crafter_ppl.py` extends `StableVideoDiffusionPipeline` with chunked encoding
- **FFmpeg subprocess encoding:** Video output piped directly to FFmpeg via stdin (`core/common/video_io.py`)
- **GPU memory management:** Explicit `release_cuda_memory()` calls between processing stages (`core/common/gpu_utils.py`)

---

*Stack analysis: 2026-04-07*
