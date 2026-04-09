# Architecture

**Analysis Date:** 2026-04-09

## Pattern Overview

**Overall:** Multi-application monolith with layered architecture

StereoCrafter is a collection of standalone GUI/CLI applications for stereo (3D) video processing. Each application is a separate entry point (`*_gui.py` files) that shares a common `core/` library. The architecture follows a **Model-View-Controller (MVC)** pattern within each application, with a shared service layer in `core/`.

**Key Characteristics:**
- Multiple independent GUI applications (Tkinter-based) sharing a common `core/` library
- Two PySide6 (Qt) alternatives: `splatting_gui_qt.py` (primary) and `depthcrafter_gui_seg.py` with Qt widgets
- Headless CLI entry point (`splat_cli.py`)
- Batch processing with background threading and queue-based progress reporting
- GPU-accelerated video processing via PyTorch/CUDA
- Sidecar-based configuration persistence (JSON files alongside media)
- Preview scaling with percentage options ("Auto" + 250% → 25%)
- Debug logging toggleable via menu
- DNxHR encoding with multiple profile options (DNxH-LB, DNxH-SQ, DNxH-HX, DNxH-HQS, DNxH-444)

## Layers

**1. Entry Point / Application Layer** - Top-level scripts
- Location: `*.py` at project root
- Contains: `splatting_gui.py`, `merging_gui.py`, `inpainting_gui.py`, `depthcrafter_gui_seg.py`, `splatting_gui_qt.py`, `splat_cli.py`
- Depends on: `core/` packages, `pipelines/`, `megaflow/`, `depthcrafter/`
- Used by: End users via batch files (`_RUN_*.bat`)

**2. Controller / Orchestration Layer** - Business logic coordinators
- Location: `core/splatting/controller.py`, `core/ui/preview_controller.py`
- Contains: `SplattingController`, `PreviewController` - manage state, batch lifecycles, progress queues
- Depends on: `core/common/`, `core/splatting/` modules
- Used by: GUI classes and CLI

**3. Service / Processing Layer** - Core algorithms
- Location: `core/splatting/` (12 modules), `core/common/` (7 modules)
- Contains: `RenderProcessor`, `BatchProcessor`, `ForwardWarpStereo`, `BorderScanner`, `VideoIO`, `SidecarConfigManager`
- Depends on: PyTorch, decord, FFmpeg, OpenCV
- Used by: Controllers and GUIs

**4. UI / Presentation Layer** - GUI components
- Location: `core/ui/` (8 modules + `workers/` subpackage)
- Contains: `ThemeManager`, `VideoPreviewer`, `PreviewCanvasWindow`, `EncodingSettingsDialog`, `QtEncodingSettingsDialog`, `PreviewRenderWorker`, `PlaybackWorker`
- Depends on: Tkinter/ttk, PySide6, PIL
- Used by: Application entry points

**5. Pipeline / ML Layer** - Deep learning models
- Location: `pipelines/`, `megaflow/`, `depthcrafter/`
- Contains: `StableVideoDiffusionInpaintingPipeline`, `MegaFlow` model, DepthCrafter UNet
- Depends on: `diffusers`, `transformers`, PyTorch
- Used by: `inpainting_gui.py`, `depthcrafter_gui_seg.py`

## Data Flow

### Splatting Pipeline (Primary Flow)
1. **Input Discovery** - `SplattingController.find_matching_pairs()` scans source/depth directories for matching video pairs
2. **Configuration** - Sidecar JSON files loaded via `SidecarConfigManager` (convergence, disparity, gamma, borders)
3. **Preview** - `PreviewController.get_frame()` reads source + depth frames → `PreviewRenderer.render_preview_frame()` → cached in `PreviewFrameBuffer` → displayed in GUI
4. **Batch Processing** - `BatchProcessor.run_batch_process()` iterates videos → `RenderProcessor.render_video()` performs forward warp on GPU → FFmpeg pipe encodes output
5. **Output** - Multi-resolution outputs (full + low-res) written to `output_splatted/` directory

### Depth Generation Flow
1. `DepthCrafterDemo` processes video segments via `depthcrafter/depth_crafter_ppl.py`
2. Depth maps saved as MP4/NPZ/EXR sequences
3. Segments merged via `depthcrafter/merge_depth_segments.py`

### Inpainting Flow
1. `StableVideoDiffusionInpaintingPipeline` (diffusers-based) processes occlusion masks
2. Reads splatted output + masks from `output_splatted/`
3. Writes inpainted results to `completed_output/`

### Merging Flow
1. `MergingGUI` combines inpainted output with original source
2. Applies color transfer, shadow blur, border padding
3. Encodes final SBS (Side-by-Side) stereo video

**State Management:**
- GUI state: Tkinter `StringVar`/`BooleanVar` with trace callbacks
- Processing state: `threading.Event` for cancellation, `queue.Queue` for progress
- Persistent state: JSON sidecar files (`.fssidecar`, `.spsidecar`) and config files (`.splatcfg`, `.json`)
- Preview caching: `PreviewFrameBuffer` with parameter-based invalidation

## Key Abstractions

**`SplattingController`** (`core/splatting/controller.py`):
- Purpose: Bridges UI/CLI with batch processing; manages project state
- Pattern: Facade over `BatchProcessor`, `SidecarConfigManager`
- Used by: `splatting_gui.py`, `splat_cli.py`

**`BatchProcessor`** (`core/splatting/batch_processing.py`):
- Purpose: Orchestrates multi-video, multi-resolution batch processing
- Pattern: Worker with `ProcessingTask` dataclasses and `ProcessingSettings` config
- Uses: `RenderProcessor` for per-video rendering

**`RenderProcessor`** (`core/splatting/render_processor.py`):
- Purpose: Core splatting algorithm - forward warp + FFmpeg encoding
- Pattern: Single-responsibility processor for one video task
- Uses: `ForwardWarpStereo` for GPU warping, FFmpeg pipes for encoding

**`PreviewController`** (`core/ui/preview_controller.py`):
- Purpose: Headless preview state management for Qt GUI
- Pattern: MVC Controller with frame caching and sidecar persistence
- Used by: `splatting_gui_qt.py`

**`SidecarConfigManager`** (`core/common/sidecar_manager.py`):
- Purpose: Centralized sidecar JSON read/write with typed defaults
- Pattern: Schema-based config manager with `SIDECAR_KEY_MAP`
- Used by: All GUIs and controllers

**`VideoIO`** (`core/common/video_io.py`):
- Purpose: Video reading/writing abstraction over decord + FFmpeg
- Pattern: Static utility class with `VideoReader` wrappers
- Provides: `read_video_frames()`, `FFmpegDepthPipeReader`, encoding pipes

**`ForwardWarpStereo`** (`core/splatting/forward_warp.py`):
- Purpose: PyTorch `nn.Module` for disparity-based image warping
- Pattern: Neural network module for GPU-accelerated stereo projection

## Entry Points

**`splatting_gui.py`** - Primary Tkinter GUI for batch splatting (stereo video generation)
- Triggers: `_RUN_Splatting_GUI.bat`
- Class: `SplatterGUI(ThemedTk)` - ~5897 lines
- Responsibilities: Video preview, parameter tuning, batch processing, sidecar management

**`splatting_gui_qt.py`** - PySide6 alternative GUI for splatting
- Triggers: `_RUN_Splatting_qt_GUI.bat`
- Class: `SplattingApp(QMainWindow)` - ~1037 lines
- Responsibilities: Interactive preview with wigglegram, real-time rendering, batch processing, sidecar management, preview scaling, debug logging toggle
- Uses: `Ui_MainWindow` (from `core/ui/splatting_ui.py`, Qt Designer generated), `PreviewController`, `PreviewRenderWorker`, `PlaybackWorker`, `SplattingController`, `ProcessingSettings`, `QtEncodingSettingsDialog`

**`merging_gui.py`** - Tkinter GUI for merging inpainted + original video
- Triggers: `_RUN_Merging_GUI.bat`
- Class: `MergingGUI(ThemedTk)` - ~2453 lines
- Responsibilities: Video merging, color transfer, shadow effects, final encoding

**`inpainting_gui.py`** - Tkinter GUI for AI-based inpainting
- Triggers: `_RUN_Inpainting_GUI.bat`
- Class: `InpaintingGUI(ThemedTk)` - ~3935 lines
- Responsibilities: Diffusion-based inpainting, tile processing, mask handling

**`depthcrafter_gui_seg.py`** - Tkinter GUI for depth map generation
- Triggers: `_RUN_DepthCrafter_GUI_Seg.bat`
- Class: DepthCrafter GUI with segmentation support - ~3026 lines
- Responsibilities: Depth estimation, segment merging, depth visualization

**`splat_cli.py`** - Headless CLI for batch splatting
- Triggers: Direct `python splat_cli.py`
- Uses: `argparse` → `SplattingController` → `BatchProcessor`
- Responsibilities: Non-interactive batch processing

**`splat_cli_test_simple.ipynb`** - Jupyter notebook for testing

## Module Boundaries

**`core/` is the shared library** - All GUIs import from `core/` but `core/` never imports from GUI files.

**`core/common/`** - Cross-cutting utilities used by all applications:
- `video_io.py` - Video I/O (decord, FFmpeg pipes)
- `sidecar_manager.py` - Sidecar JSON config
- `gpu_utils.py` - CUDA memory management
- `encoding_utils.py` - FFmpeg encoder configuration
- `image_processing.py` - OpenCV/torch image ops (blur, dilate, anaglyph, color transfer)
- `file_organizer.py` - Move-to-finished workflow
- `cli_utils.py` - Logging, progress bar drawing

**`core/splatting/`** - Splatting-specific modules (only used by splatting GUIs/CLI):
- `controller.py` - Orchestration
- `batch_processing.py` - Batch worker
- `render_processor.py` - Per-video rendering
- `forward_warp.py` - GPU warping
- `depth_processing.py` - Depth normalization/gamma
- `convergence.py` - Auto-convergence estimation
- `border_scanning.py` - Border analysis
- `config_manager.py` - App config persistence
- `fusion_export.py` - DaVinci Fusion sidecar generation
- `m2s_mask.py` - Occlusion mask building
- `preview_rendering.py` - Preview frame rendering
- `analysis_service.py` - Depth statistics computation

**`core/ui/`** - UI components (used by Tkinter GUIs):
- `widgets.py` - Tooltip, slider helpers
- `video_previewer.py` - Video list navigation
- `theme_manager.py` - Dark/light theming
- `preview_canvas_window.py` - Preview window
- `preview_buffer.py` - Frame caching
- `encoding_settings.py` - Encoding dialog
- `dnd_support.py` - Drag-and-drop support
- `workers/` - Background workers (`preview_render_worker.py`, `playback_worker.py`)

**`pipelines/`** - ML pipeline (used by inpainting GUI only):
- `stereo_video_inpainting.py` - Diffusers-based SVD inpainting pipeline

**`megaflow/`** - Optical flow model (vendored):
- `model/megaflow.py` - Main model
- `model/layers/` - ViT building blocks
- Used for: Motion estimation (not directly invoked by GUIs currently)

**`depthcrafter/`** - Depth estimation (vendored):
- `depth_crafter_ppl.py` - Pipeline
- `depthcrafter_logic.py` - Demo logic
- `merge_depth_segments.py` - Segment merging
- `unet.py` - UNet architecture

**`dependency/`** - Third-party vendored code:
- `forward_warp_pytorch/` - Forward warp implementations
- `u2net.py` - U2Net model
- `convergence_estimator.py` - Legacy convergence estimation

**`benchmark/`** - Evaluation tools:
- `infer/infer_batch.py` - Batch inference
- `eval/eval.py` - Metric evaluation
- `dataset_extract/` - Dataset extractors (KITTI, NYU, Sintel, ScanNet, Bonn)

**`scripts/Davinci Resolve/`** - Resolve integration scripts:
- Marker export/import, keyframe generation

**Config files** at root:
- `config_depthcrafter.json` - DepthCrafter GUI config
- `config_inpaint.json` - Inpainting GUI config
- `config_merging.mergecfg` - Merging GUI config
- `config_splat.splatcfg` - Splatting GUI config

## Error Handling

**Strategy:** Logging-based with GUI feedback via progress queues

**Patterns:**
- `try/except` with `logger.error()`/`logger.warning()` throughout
- GUI progress queue messages: `("status", "message")` tuples
- `threading.Event` for graceful cancellation (`stop_event`)
- FFmpeg pipe error reading via background thread (`_read_ffmpeg_output`)
- Import fallbacks with `try/except ImportError` for optional dependencies

## Cross-Cutting Concerns

**Logging:** Python `logging` module; per-module loggers via `logging.getLogger(__name__)`; configurable level via `set_logger_level()` in `core/common/cli_utils.py`

**Validation:** Manual type checking in sidecar manager (`SIDECAR_KEY_MAP` with expected types); Qt validators (`QIntValidator`, `QDoubleValidator`)

**Threading:** Background processing via `threading.Thread` (daemon); progress via `queue.Queue`; cancellation via `threading.Event`; Qt uses `QThread` + signals/slots

**GPU Management:** `core/common/gpu_utils.py` provides `check_cuda_availability()`, `release_cuda_memory()`; explicit `gc.collect()` calls between heavy operations

**Encoding:** Centralized in `core/common/encoding_utils.py` with support for H.264, H.265, and DNxHR profiles (DNxH-LB, DNxH-SQ, DNxH-HX, DNxH-HQS, DNxH-444); configurable CRF, quality presets, NVENC options; dynamic container/encoder options based on selected codec

---

*Architecture analysis: 2026-04-09*