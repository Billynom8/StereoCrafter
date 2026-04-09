# External Integrations

**Analysis Date:** 2026-04-09

## Model Hub / External Services

**HuggingFace Hub:**
- **Purpose:** Download pre-trained model weights at runtime
- **SDK:** `huggingface-hub` (>=1.7.1)
- **Used in:**
  - `megaflow/megaflow_masker.py` — `hf_hub_download()` for MegaFlow optical flow checkpoints
  - `megaflow/model/megaflow.py` — `MegaFlow.from_pretrained()` downloads from HuggingFace repos
  - `pipelines/stereo_video_inpainting.py` — `from_pretrained()` for CLIP, VAE, UNet components
  - `depthcrafter/depthcrafter_logic.py` — `from_pretrained()` for DepthCrafter UNet and pipeline
- **Models referenced:**
  - `stabilityai/stable-video-diffusion-img2vid-xt-1-1` — SVD base model (local in `weights/`)
  - `TencentARC/StereoCrafter` — StereoCrafter inpainting weights (local in `weights/`)
  - `tencent/DepthCrafter` — DepthCrafter depth estimation weights (local in `weights/`)
  - `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` — CLIP image encoder (loaded via SVD pipeline)
  - `megaflow-flow` / `megaflow-track` — MegaFlow optical flow models
- **Auth:** None required (public models); `HF_TOKEN` env var supported by huggingface-hub if needed
- **Offline mode:** `use_local_models_only_var` flag in `config_depthcrafter.json`

## Data Storage

**File System Only:**
- No database detected
- Configuration stored as JSON files:
  - GUI configs: `config_depthcrafter.json`, `config_inpaint.json`
  - Per-video sidecars: `.fssidecar` files co-located with video outputs (`core/common/sidecar_manager.py`)
  - Help content: `dependency/inpaint_help.json`, `dependency/merge_help.json`, `dependency/splatter_help.json`, `depthcrafter/help_content.json`

**Model Weights:**
- Local directory: `weights/`
  - `weights/stable-video-diffusion-img2vid-xt-1-1/` — SVD model
  - `weights/DepthCrafter/` — DepthCrafter model
  - `weights/StereoCrafter/` — StereoCrafter model
- External model: `dependency/iw3_sod_v1_20260122.pth` — U2Net salient object detection

**Caching:**
- Frame buffer caching in `core/ui/preview_buffer.py` (in-memory, per-session)
- Convergence cache in `core/splatting/convergence_cache.py` (in-memory)
- `.ruff_cache/` — Ruff lint cache

## Authentication

**None detected.** The application is fully local/offline-capable. All model weights are either:
- Pre-downloaded to `weights/` directory
- Downloaded from public HuggingFace repos (no auth required)

No API keys, tokens, or credential files detected in the codebase.

## Third-Party Services

**FFmpeg (External System Tool):**
- **Purpose:** Video encoding/decoding, depth map extraction
- **Used in:**
  - `core/common/video_io.py` — `start_ffmpeg_pipe_process()`, `FFmpegRGBSingleFrameReader`, `FFmpegDepthPipeReader`
  - `core/ui/preview_controller.py` — `_read_depth_frame_ffmpeg()` for 10-bit+ depth maps
  - `core/ui/video_previewer.py` — FFmpeg-based depth frame decoding
  - `depthcrafter/utils.py` — `mediapy` with `ffmpeg_args` for EXR/HEVC output
  - `merging_gui.py` — Direct FFmpeg pipe for final video encoding
- **Encoding support:** H.264, H.265, CPU (libx264/libx265) and GPU NVENC (`core/common/encoding_utils.py`)
- **Configuration:** Quality presets (Fastest→Slowest), CRF, NVENC settings (lookahead, AQ)

**CUDA / NVIDIA:**
- **Purpose:** GPU acceleration for all ML inference
- **Components:** CUDA 12.8, xFormers, Triton, NVENC encoding
- **Used in:** All model inference, forward warping, depth processing

**Decord:**
- **Purpose:** Fast video reading (alternative to FFmpeg)
- **Used in:** `core/common/video_io.py`, `depthcrafter_gui_seg.py`, `inpainting_gui.py`, `merging_gui.py`

**OpenEXR / Imath:**
- **Purpose:** High-bit-depth depth map I/O (10-bit, 16-bit)
- **Used in:** `depthcrafter_gui_seg.py`, `tools/npz_to_exr.py`
- **Graceful degradation:** Features disabled if libraries not found

## Data Flow

### Stereo Video Generation Pipeline

1. **Input:** Monocular video file(s)
2. **Depth Estimation** (`depthcrafter_gui_seg.py`):
   - Load SVD + DepthCrafter models from `weights/`
   - Process video in segments (windowed inference)
   - Output: Depth maps as NPZ/EXR/MP4/PNG
3. **Depth Merging** (`depthcrafter/merge_depth_segments.py`):
   - Merge segment depth maps into full-video depth
   - Optional: percentile normalization, gamma correction, dithering
4. **Splatting** (`core/splatting/`):
   - Load depth maps + source video
   - Forward warp to create right-eye view (`core/splatting/forward_warp.py`)
   - Output: Splatted video grid (source + depth + mask + right eye)
5. **Inpainting** (`inpainting_gui.py`):
   - Load SVD + StereoCrafter models
   - Inpaint splatted artifacts using diffusion
   - Output: Clean stereo video (SBS + anaglyph)
6. **Merging** (`merging_gui.py`):
   - Combine inpainted output with original video
   - Apply masks, borders, color transfer, shadow effects
   - Encode final output via FFmpeg pipe

### Preview Pipeline (GUI)

1. `PreviewController` loads video + depth map
2. Frame buffer caching (`core/ui/preview_buffer.py`)
3. Background rendering via `PreviewRenderWorker` (QThread)
4. Display in PySide6 or Tkinter preview window

## Configuration

**GUI Config Files:**
- `config_depthcrafter.json` — DepthCrafter settings
- `config_inpaint.json` — Inpainting settings
- `config_merging.mergecfg` — Merging settings
- `config_splat.splatcfg` — Splatting settings

**Sidecar Files:**
- Per-video JSON files storing splatting parameters (convergence, disparity, gamma, etc.)
- Managed by `core/common/sidecar_manager.py`
- Key map defined in `SidecarConfigManager.SIDECAR_KEY_MAP`

**Model Loading:**
- Models loaded from local `weights/` directory by default
- HuggingFace fallback if `use_local_models_only_var` is false
- MegaFlow models downloaded on-demand via `hf_hub_download()`

**Environment Variables:**
- No required env vars detected
- CUDA device selection handled internally
- `HF_TOKEN` optionally supported by huggingface-hub for private models

## Webhooks & Callbacks

**None detected.** The application is entirely local with no incoming or outgoing webhooks.

---

*Integration audit: 2026-04-07*
