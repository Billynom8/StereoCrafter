# Coding Conventions

**Analysis Date:** 2026-04-09

## Code Style

**Formatter:**
- **Ruff** - Configured in `pyproject.toml` (lines 47-61)
- Line length: 120 characters
- Indent width: 4 spaces
- Indent style: space
- Skip magic trailing comma: true

**Linter:**
- **Ruff** - Selects `E`, `F`, `W` (pycodestyle errors/warnings + pyflakes)
- Ignores `E501` (line length - handled by formatter)
- Unfixable: `F841` (unused variables - left for manual review)

**Config:** `pyproject.toml`

## Naming Patterns

**Files:**
- Snake case throughout: `video_io.py`, `image_processing.py`, `preview_controller.py`
- GUI entry points use descriptive names: `splatting_gui_qt.py`, `merging_gui.py`, `inpainting_gui.py`
- Config files use project prefix: `config_depthcrafter.json`, `config_inpaint.json`

**Variables:**
- Snake case: `video_path`, `total_frames`, `max_disp`, `use_gpu`
- Private/protected: underscore prefix `_proc`, `_next_frame`, `_force_fallback`, `_params_signature`
- Constants: UPPER_SNAKE_CASE: `CUDA_AVAILABLE`, `DARK_COLORS`, `LIGHT_COLORS`, `DEFAULT_ENCODING_CONFIG`
- Module-level config flags: `_ENABLE_XFORMERS_ATTENTION` in `depthcrafter/depthcrafter_logic.py`

**Functions:**
- Snake case with descriptive verbs: `read_video_frames`, `apply_mask_dilation`, `build_encoder_args`
- Boolean-returning functions: `is_dark_mode()`, `is_depth_high_bit()`, `is_playing()`
- Private helpers: underscore prefix `_build_cmd()`, `_ensure_process()`, `_read_exact()`, `_probe_depth_properties()`

**Classes:**
- PascalCase: `VideoIO`, `FFmpegRGBPipeReader`, `PreviewController`, `SplattingApp`, `ThemeManager`
- Private/internal classes: underscore prefix `_NumpyBatch`, `_ResizingDepthReader`
- Workers: `PreviewRenderWorker`, `PlaybackWorker`, `FileOrganizerWorker`

**Types:**
- PascalCase for type aliases when used (rare - mostly uses inline typing)
- Dataclasses used in `pipelines/stereo_video_inpainting.py` (`StableVideoDiffusionPipelineOutput`)

## Import Patterns

**Order (observed in `core/common/video_io.py`):**
1. Standard library imports: `os`, `re`, `json`, `shutil`, `threading`, `time`, `logging`, `subprocess`
2. Third-party imports: `numpy as np`, `torch`, `cv2`, `from decord import VideoReader, cpu`
3. Local imports: `from core.common.gpu_utils import CUDA_AVAILABLE`

**Import style:**
- Absolute imports for core modules: `from core.common.video_io import ...`
- Relative imports within packages: `from .batch_processing import ...` in `core/splatting/__init__.py`
- `as` aliases for common libraries: `import numpy as np`, `import cv2` (no alias), `import torch` (no alias)
- Wildcard imports avoided; explicit imports preferred

**Path aliases:**
- No path aliases configured; all imports use full package paths from project root
- `core` package is the primary internal import root

## Error Handling

**Patterns:**
- **Logging-based error reporting** - `logger.error()`, `logger.warning()`, `logger.debug()` throughout
  - Example: `core/common/video_io.py` line 901: `logger.error(f"Encoding failed: {e}")`
- **Exception suppression with fallback** - Broad `except Exception:` with graceful degradation
  - Example: `core/common/video_io.py` lines 627-636: Falls back to ffprobe if decord fails
- **Return None/False on failure** - Functions return `None` or `False` instead of raising
  - Example: `core/splatting/controller.py` line 44: `return []` for invalid input
  - Example: `core/ui/preview_controller.py` line 327: `return None` when reader is not initialized
- **Bare `except Exception: pass`** - Common in UI code for widget configuration
  - Example: `core/ui/theme_manager.py` lines 204-205, 227-228, 239-240: `except Exception: pass`
- **EOFError for stream termination** - Used in FFmpeg readers
  - Example: `core/common/video_io.py` line 194: `raise EOFError("FFmpegRGBPipeReader reached EOF...")`
- **ValueError for invalid input** - Example: `core/common/video_io.py` line 638: `raise ValueError(f"Could not determine video dimensions...")`
- **NotImplementedError for unsupported features** - Example: `core/common/video_io.py` line 606: `raise NotImplementedError(f"Dataset '{dataset}' not supported.")`

**Anti-pattern observed:** Broad `except Exception: pass` blocks silently swallow errors in UI styling code (`core/ui/theme_manager.py`, `core/ui/widgets.py`).

## Type System

**Typing approach:** Mixed, leaning toward gradual typing

- **Type hints used** for function signatures: `def read_video_frames(...) -> Tuple[Any, float, int, int, int, int, Optional[dict], int]`
- **`typing` module** heavily used: `Optional`, `Tuple`, `Dict`, `List`, `Any`, `Union`, `Callable`
- **No type checker** enforced (no mypy config, no pyright config)
- **`TYPE_CHECKING`** used selectively: `core/ui/theme_manager.py` line 8: `from typing import TYPE_CHECKING`
- **No runtime type validation** - Type hints are purely informational
- **Docstring type annotations** complement type hints in Args/Returns sections

## Common Patterns

**Module-level logger:**
```python
logger = logging.getLogger(__name__)
```
Seen in virtually every module: `core/common/video_io.py`, `core/ui/theme_manager.py`, `core/splatting/controller.py`

**`__init__.py` barrel exports:**
```python
from .video_io import VideoIO, read_video_frames
from .file_organizer import move_files_to_finished, ...
__all__ = ["VideoIO", "read_video_frames", ...]
```
Seen in `core/__init__.py`, `core/common/__init__.py`, `core/ui/__init__.py`, `core/splatting/__init__.py`

**GPU/CPU dual-path execution:**
```python
if use_gpu and mask.is_cuda:
    # GPU path using torch operations
else:
    # CPU fallback using cv2/numpy
```
Seen in `core/common/image_processing.py` (multiple functions)

**FFmpeg subprocess piping:**
- Pattern: `subprocess.Popen` with `stdout=subprocess.PIPE` for raw video streams
- Seen in `core/common/video_io.py`: `FFmpegRGBPipeReader`, `FFmpegDepthPipeReader`

**Progress queue + stop event threading:**
```python
self.progress_queue = queue.Queue()
self.stop_event = threading.Event()
```
Seen in `core/splatting/controller.py`

**Sidecar JSON config persistence:**
- Pattern: `SidecarConfigManager` for loading/saving per-video settings as JSON
- Seen in `core/common/sidecar_manager.py`, `core/ui/preview_controller.py`

**Parameter mapping/dictionary-based config:**
- Functions accept `Dict[str, Any]` params dictionaries rather than typed config objects
- Seen in `core/ui/preview_controller.py` `get_frame(params: Dict[str, Any])`

**Docstring convention:** Google-style docstrings with Args/Returns sections
```python
"""Short description.

Args:
    param_name: Description

Returns:
    Description of return value
"""
```

## Anti-Patterns Observed

1. **Bare `except Exception: pass`** - Silent error swallowing in UI code (`core/ui/theme_manager.py`, `core/ui/widgets.py`)
2. **Large files** - `core/common/video_io.py` (1124 lines), `megaflow/model/megaflow.py` (1023 lines)
3. **Global mutable state** - `_FFPROBE_AVAIL`, `_INFO_CACHE`, `CUDA_AVAILABLE` in `core/common/video_io.py` and `core/common/gpu_utils.py`
4. **Inconsistent logger naming** - `_logger` in `depthcrafter/depthcrafter_logic.py` vs `logger` everywhere else
5. **TODO comments** - 3 found: `core/ui/preview_controller.py:266`, `pipelines/stereo_video_inpainting.py:596`, `depthcrafter/unet.py:24`
6. **No test infrastructure** - Zero test files, no pytest/unittest config, no conftest.py
7. **Mixed GUI frameworks** - Both tkinter (`splatting_gui.py`, `merging_gui.py`) and PySide6/Qt (`splatting_gui_qt.py`) coexist
8. **Duplicate functionality** - `read_video_frames` exists in both `core/common/video_io.py` and `depthcrafter/utils.py`

---

*Convention analysis: 2026-04-07*
