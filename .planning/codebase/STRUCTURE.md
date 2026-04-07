# Codebase Structure

**Analysis Date:** 2026-04-07

## Directory Layout

```
StereoCrafter/
├── core/                    # Shared library (used by all GUIs/CLI)
│   ├── __init__.py          # Package exports (VideoIO, ThemeManager, etc.)
│   ├── common/              # Cross-cutting utilities
│   │   ├── __init__.py      # Exports: VideoIO, read_video_frames, file organizer
│   │   ├── video_io.py      # Video I/O (decord, FFmpeg pipes, encoding)
│   │   ├── sidecar_manager.py # Sidecar JSON config management
│   │   ├── gpu_utils.py     # CUDA availability, memory release
│   │   ├── encoding_utils.py # FFmpeg encoder configuration
│   │   ├── image_processing.py # OpenCV/torch image ops
│   │   ├── file_organizer.py  # Move-to-finished workflow
│   │   ├── cli_utils.py       # Logging setup, progress bar
│   │   └── mesh_warp.py       # Mesh warping utilities
│   ├── ui/                  # UI components (Tkinter/Qt)
│   │   ├── __init__.py      # Exports: ThemeManager, PreviewCanvasWindow, etc.
│   │   ├── widgets.py       # Tooltip, slider helpers
│   │   ├── video_previewer.py # Video list navigation widget
│   │   ├── theme_manager.py   # Dark/light theming
│   │   ├── preview_canvas_window.py # Preview window
│   │   ├── preview_buffer.py    # Frame caching
│   │   ├── preview_controller.py # Qt preview controller
│   │   ├── encoding_settings.py  # Encoding dialog
│   │   ├── dnd_support.py        # Drag-and-drop support
│   │   ├── splatting_ui.py       # Qt UI definition (generated)
│   │   └── workers/              # Background workers
│   │       ├── __init__.py
│   │       ├── preview_render_worker.py # Qt render worker
│   │       └── playback_worker.py       # Qt playback worker
│   └── splatting/           # Splatting-specific modules
│       ├── __init__.py      # Exports all splatting components
│       ├── controller.py    # SplattingController (orchestration)
│       ├── batch_processing.py # BatchProcessor, ProcessingSettings
│       ├── render_processor.py # RenderProcessor (per-video rendering)
│       ├── forward_warp.py  # ForwardWarpStereo (PyTorch nn.Module)
│       ├── depth_processing.py # Depth normalization, gamma, stats
│       ├── convergence.py   # Auto-convergence estimation
│       ├── border_scanning.py # Border analysis
│       ├── config_manager.py  # App config persistence
│       ├── fusion_export.py   # DaVinci Fusion sidecar generation
│       ├── m2s_mask.py        # Occlusion mask building
│       ├── preview_rendering.py # Preview frame rendering
│       ├── analysis_service.py  # Depth statistics computation
│       └── convergence_cache.py # Convergence result caching
├── pipelines/               # ML pipelines
│   ├── __init__.py
│   └── stereo_video_inpainting.py # Diffusers SVD inpainting
├── megaflow/                # Optical flow model (vendored)
│   ├── __init__.py          # Exports: MegaFlow
│   ├── megaflow_masker.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── basic.py
│   └── model/
│       ├── __init__.py
│       ├── megaflow.py      # Main model
│       ├── layer.py
│       ├── refine.py
│       ├── model_utils.py
│       ├── matching.py
│       ├── geometry.py
│       ├── flow_head.py
│       └── layers/           # ViT building blocks
│           ├── __init__.py
│           ├── attention.py
│           ├── block.py
│           ├── vision_transformer.py
│           ├── swiglu_ffn.py
│           ├── rope.py
│           ├── patch_embed.py
│           ├── mlp.py
│           ├── layer_scale.py
│           └── drop_path.py
├── depthcrafter/            # Depth estimation (vendored)
│   ├── __init__.py
│   ├── depth_crafter_ppl.py   # Pipeline
│   ├── depthcrafter_logic.py  # Demo logic
│   ├── merge_depth_segments.py # Segment merging
│   ├── dav_util.py            # DaVinci Resolve utilities
│   ├── utils.py               # Helper functions
│   ├── unet.py                # UNet architecture
│   └── help_content.json      # GUI help texts
├── dependency/              # Third-party vendored code
│   ├── forward_warp_pytorch/  # Forward warp implementations
│   │   ├── __init__.py
│   │   ├── forward_warp.py
│   │   ├── forward_warp_pytorch.py
│   │   ├── forward_warp_rescaled.py
│   │   └── forward_warp_max_motion_pytorch.py
│   ├── u2net.py               # U2Net model
│   ├── convergence_estimator.py # Legacy convergence estimation
│   ├── splatter_help.json     # Splatting GUI help content
│   ├── merge_help.json        # Merging GUI help content
│   └── inpaint_help.json      # Inpainting GUI help content
├── benchmark/               # Evaluation tools
│   ├── __init__.py
│   ├── infer/
│   │   └── infer_batch.py   # Batch inference
│   ├── eval/
│   │   ├── eval.py          # Metric evaluation
│   │   └── metric.py        # Metric implementations
│   └── dataset_extract/      # Dataset extractors
│       ├── dataset_extract_kitti.py
│       ├── dataset_extract_nyu.py
│       ├── dataset_extract_sintel.py
│       ├── dataset_extract_scannet.py
│       └── dataset_extract_bonn.py
├── scripts/                 # Utility scripts
│   └── Davinci Resolve/     # Resolve integration
│       ├── Edit/
│       │   ├── Export_Markers_To_SRT.py
│       │   └── Add Markers.py
│       └── Comp/
│           ├── Markers to Keyframes.py
│           └── Control Data Exporter.py
├── visualization/           # Visualization tools
│   └── visualization_pcd.py  # Point cloud visualization
├── tools/                   # Conversion tools
│   └── npz_to_exr.py        # NPZ to EXR converter
├── old/                     # Archived/legacy code
├── _install/                # Installation scripts
├── assets/                  # Static assets (images, icons)
├── source_video/            # Sample/test videos
├── weights/                 # Model weights directory
├── splatting_gui.py         # Primary Tkinter GUI (splatting) ~4000+ lines
├── splatting_gui_qt.py      # PySide6 GUI (splatting) 574 lines
├── merging_gui.py           # Tkinter GUI (merging) ~2453 lines
├── inpainting_gui.py        # Tkinter GUI (inpainting) ~3935 lines
├── depthcrafter_gui_seg.py  # Tkinter GUI (depth generation) ~3026 lines
├── splat_cli.py             # Headless CLI for splatting 175 lines
├── splat_cli_test_simple.ipynb # Jupyter test notebook
├── pyproject.toml           # Project config + dependencies (uv)
├── uv.lock                  # Lockfile (uv)
├── config_depthcrafter.json # DepthCrafter GUI config
├── config_inpaint.json      # Inpainting GUI config
├── config_merging.mergecfg  # Merging GUI config
├── config_splat.splatcfg    # Splatting GUI config
├── .python-version          # Python version pin
├── _RUN_DepthCrafter_GUI_Seg.bat # Windows launcher
├── _RUN_Inpainting_GUI.bat  # Windows launcher
├── _RUN_Merging_GUI.bat     # Windows launcher
├── _RUN_Splatting_GUI.bat   # Windows launcher
├── _RUN_Splatting_qt_GUI.bat # Windows launcher
├── _update.bat              # Update script
├── README.md                # Project documentation
├── PROGRESS.md              # Development progress tracking
├── .gitignore
└── .ruff_cache/             # Ruff linter cache
```

## Key Directories

- **`core/`** - Shared library containing all reusable modules. Three subpackages: `common/` (cross-cutting), `ui/` (presentation), `splatting/` (domain-specific). This is the primary location for new shared code.

- **`core/common/`** - Utilities used across all GUI applications. Add new shared helpers here (video I/O, image processing, file management, GPU utils).

- **`core/splatting/`** - Splatting domain logic. Add new splatting algorithms, processors, or services here. Each module has a single responsibility (e.g., `forward_warp.py` for warping, `border_scanning.py` for border analysis).

- **`core/ui/`** - Reusable UI components. Add new widgets, dialogs, or theme-related code here. The `workers/` subpackage contains Qt-specific background workers.

- **`pipelines/`** - ML pipeline implementations. Currently contains the diffusers-based inpainting pipeline. Add new ML pipelines here.

- **`megaflow/`** - Vendored optical flow model. Treat as external dependency - do not modify unless updating the vendored version.

- **`depthcrafter/`** - Vendored depth estimation model. Treat as external dependency.

- **`dependency/`** - Vendored third-party code (forward warp implementations, U2Net). Treat as external.

- **`benchmark/`** - Evaluation and dataset tools. Not used by GUIs.

- **`scripts/`** - External integration scripts (DaVinci Resolve).

## Key Files

- **`pyproject.toml`** - Project configuration, dependencies, uv package manager settings, ruff linting config. All dependency management goes here.

- **`core/__init__.py`** - Top-level package exports. Controls what's publicly available from `core`.

- **`core/splatting/__init__.py`** - Splatting package exports. Re-exports all splatting components for convenient imports.

- **`core/splatting/controller.py`** - Main orchestration controller. Entry point for all splatting operations from GUI/CLI.

- **`core/splatting/batch_processing.py`** - Batch processing engine (~1215 lines). Contains `ProcessingTask` and `ProcessingSettings` dataclasses.

- **`core/splatting/render_processor.py`** - Core rendering loop (~693 lines). The actual splatting algorithm.

- **`core/common/video_io.py`** - Video I/O utilities (~1124 lines). Primary interface for reading/writing video.

- **`core/common/sidecar_manager.py`** - Sidecar configuration (~231 lines). Central schema for per-clip settings.

- **`splatting_gui.py`** - Primary application (~5897 lines). Main Tkinter GUI with all features.

- **`splatting_gui_qt.py`** - Qt alternative (574 lines). Cleaner architecture using `PreviewController`.

## Naming Conventions

**Files:**
- `snake_case.py` for all Python modules
- `_RUN_*.bat` for Windows launchers (prefixed with underscore)
- `config_*.json` / `config_*.<ext>` for application config files
- `*_gui.py` for GUI entry points
- `*_cli.py` for CLI entry points
- `*.splatcfg`, `*.mergecfg` for custom config formats
- `*.fssidecar`, `*.spsidecar` for sidecar files (generated at runtime)

**Directories:**
- `snake_case` for all directories
- `__init__.py` in all packages (some empty, some with exports)

**Classes:**
- `PascalCase` for all classes (`SplatterGUI`, `SplattingController`, `RenderProcessor`)
- Suffix patterns: `*GUI`, `*Controller`, `*Processor`, `*Manager`, `*Worker`, `*Service`

**Functions/Methods:**
- `snake_case` for all functions and methods
- Private methods prefixed with `_` (e.g., `_on_border_mode_change`, `_load_config`)
- Event handlers prefixed with `_on_` (e.g., `_on_map_selection_changed`)

**Variables:**
- `snake_case` for module-level and local variables
- Tkinter variables use `_var` suffix (e.g., `max_disp_var`, `dark_mode_var`)
- Constants in `UPPER_SNAKE_CASE` (e.g., `GUI_VERSION`, `VIDEO_EXTS`, `DEPTH_VIS_TV10_BLACK_NORM`)

**Dataclasses:**
- `ProcessingSettings`, `ProcessingTask` - configuration carriers in `core/splatting/batch_processing.py`

## File Organization Patterns

**Package exports:** Each `__init__.py` re-exports public API for convenient imports:
```python
# core/splatting/__init__.py - re-exports all components
from .controller import SplattingController
from .batch_processing import BatchProcessor, ProcessingSettings
# ... etc
```

**Module-level constants:** GUI defaults and constants defined as class attributes on the GUI class:
```python
class SplatterGUI(ThemedTk):
    APP_CONFIG_DEFAULTS = { "MAX_DISP": "30.0", ... }
    SIDECAR_KEY_MAP = { "convergence_plane": "CONV_POINT", ... }
```

**Refactoring pattern:** Logic extracted from large GUI files into `core/` modules, with `# [REFACTORED]` comments marking the migration:
```python
# [REFACTORED] FusionSidecarGenerator class replaced with core import
from core.splatting import FusionSidecarGenerator
```

**Worker pattern:** Background operations use either:
- Tkinter: `threading.Thread` + `queue.Queue` + `after()` polling
- Qt: `QThread` + `moveToThread()` + signals/slots

## Where to Add New Code

**New shared utility:**
- Implementation: `core/common/<utility_name>.py`
- Export: Add to `core/common/__init__.py`
- Tests: (no test directory exists yet)

**New splatting feature:**
- Implementation: `core/splatting/<feature_name>.py`
- Export: Add to `core/splatting/__init__.py`
- Controller integration: `core/splatting/controller.py`

**New UI component:**
- Tkinter widget: `core/ui/<widget_name>.py`
- Qt worker: `core/ui/workers/<worker_name>.py`
- Export: Add to `core/ui/__init__.py`

**New GUI application:**
- Entry point: `<feature_name>_gui.py` at project root
- Follow pattern: Inherit from `ThemedTk`, use `core/` modules
- Launcher: `_RUN_<Feature>_GUI.bat`

**New ML pipeline:**
- Implementation: `pipelines/<pipeline_name>.py`
- Follow pattern: Inherit from `DiffusionPipeline` if using diffusers

**New config option:**
- Sidecar schema: Add to `SidecarConfigManager.SIDECAR_KEY_MAP` in `core/common/sidecar_manager.py`
- GUI default: Add to `APP_CONFIG_DEFAULTS` in the relevant GUI class
- Processing setting: Add to `ProcessingSettings` dataclass in `core/splatting/batch_processing.py`

## Special Directories

**`weights/`** - Model weights storage. Not committed to git. Downloaded at runtime or manually placed.

**`old/`** - Archived/legacy code. Not actively maintained. Reference only.

**`_install/`** - Installation/setup scripts. Windows-specific.

**`.planning/`** - GSD planning documents. Generated by tooling.

**`.venv/`** - Virtual environment (uv-managed). Not committed.

**`.ruff_cache/`** - Ruff linter cache. Not committed.

---

*Structure analysis: 2026-04-07*