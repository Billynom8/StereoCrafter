# Testing

**Analysis Date:** 2026-04-07

## Framework

**Not detected.** No testing framework is configured in this codebase.

- No `pytest`, `unittest`, `unittest2`, or `nose` configuration found
- No `conftest.py` files
- No test configuration in `pyproject.toml`
- No `pytest.ini`, `setup.cfg`, or `tox.ini`

## Test Location

**No test files found.**

- Search for `*test*`, `*spec*`, `*test.py*`, `*_test.py*` returned zero Python test files
- The only test-related files found are:
  - `splat_cli_test_simple.ipynb` - Jupyter notebook (manual testing)
  - `write_test.txt` - Unknown purpose (not executable tests)
  - `benchmark/csv/meta_nyu_test.csv` - Dataset metadata, not unit tests

## Test Structure

**Not applicable.** The codebase has no automated test infrastructure.

## Test Patterns

**Not applicable.** No test patterns exist in the codebase.

## Mocking

**Not applicable.** No mocking framework or patterns detected.

## Running Tests

**No test runner configured.**

There are no commands to run tests because no test framework is installed or configured. The `pyproject.toml` dev dependencies only include:

```toml
[dependency-groups]
dev = [
    "ruff>=0.15.0",
]
```

Ruff is configured for linting/formatting only, not testing.

## Coverage

**Not configured.** No coverage tool (pytest-cov, coverage.py) is installed or configured.

## Notable Gaps

**The entire codebase is untested.** This is a critical gap affecting all areas:

### Untested Core Logic
- `core/common/video_io.py` (1124 lines) - Video I/O, FFmpeg piping, encoding - **High risk**
- `core/common/image_processing.py` (500 lines) - GPU/CPU image operations, anaglyph transforms - **High risk**
- `core/common/encoding_utils.py` (241 lines) - FFmpeg encoder argument building - **High risk**
- `core/splatting/controller.py` (160 lines) - Batch processing orchestration - **High risk**
- `core/ui/preview_controller.py` (596 lines) - Preview state machine, rendering coordination - **High risk**
- `core/ui/theme_manager.py` (353 lines) - Theme management - **Medium risk**
- `core/ui/preview_buffer.py` (142 lines) - Frame caching - **Medium risk**

### Untested ML/Model Code
- `megaflow/model/megaflow.py` (1023 lines) - Optical flow model - **High risk**
- `depthcrafter/depthcrafter_logic.py` (850 lines) - Depth generation pipeline - **High risk**
- `pipelines/stereo_video_inpainting.py` (766 lines) - Video inpainting pipeline - **High risk**

### Untested GUI Code
- `splatting_gui_qt.py` (574 lines) - PySide6 GUI application
- `splatting_gui.py` - Tkinter GUI application
- `merging_gui.py` - Tkinter merging GUI
- `inpainting_gui.py` - Tkinter inpainting GUI
- `depthcrafter_gui_seg.py` - Tkinter depth crafter GUI

### Untested Utilities
- `core/common/gpu_utils.py` - CUDA availability detection
- `core/common/sidecar_manager.py` - Sidecar JSON persistence
- `core/common/file_organizer.py` - File organization workers
- `core/common/cli_utils.py` - CLI utilities
- `core/splatting/batch_processing.py` - Batch processing logic
- `core/splatting/config_manager.py` - Configuration management
- `core/splatting/convergence.py` - Convergence estimation
- `core/splatting/depth_processing.py` - Depth processing utilities
- `core/splatting/forward_warp.py` - Forward warping implementation
- `core/splatting/render_processor.py` - Render processing
- `core/splatting/analysis_service.py` - Analysis services
- `core/splatting/fusion_export.py` - Fusion sidecar generation
- `core/splatting/m2s_mask.py` - M2S occlusion mask building
- `core/splatting/border_scanning.py` - Border scanning logic
- `core/splatting/convergence_cache.py` - Convergence caching

### Benchmark Code (Partial Testing)
- `benchmark/infer/infer_batch.py` - Batch inference (manual execution)
- `benchmark/eval/eval.py` - Evaluation scripts (manual execution)
- `benchmark/eval/metric.py` - Metric calculations (could be unit tested)
- `benchmark/dataset_extract/*.py` - Dataset extraction scripts

## Recommended Testing Strategy

Given the complete absence of tests, a phased approach is recommended:

### Phase 1: Infrastructure Setup
1. Add `pytest` to dev dependencies in `pyproject.toml`
2. Create `tests/` directory with `conftest.py`
3. Configure pytest in `pyproject.toml` under `[tool.pytest.ini_options]`

### Phase 2: Unit Tests for Pure Functions (Highest ROI)
Start with functions that have no external dependencies (no FFmpeg, no GPU, no files):
- `core/common/encoding_utils.py` - `build_encoder_args`, `get_encoder_codec`, `quality_to_preset`
- `core/common/image_processing.py` - `apply_borders_to_frames` (pure tensor operations)
- `core/ui/preview_controller.py` - `sync_sliders_to_auto_borders` (static method)
- `core/splatting/config_manager.py` - Configuration loading/saving logic

### Phase 3: Core Logic with Mocking
- `core/common/video_io.py` - Mock FFmpeg subprocess calls
- `core/splatting/controller.py` - Mock batch processor
- `core/ui/preview_buffer.py` - Test cache invalidation logic

### Phase 4: Integration Tests
- End-to-end pipeline tests with small test video files
- GPU-dependent tests (skip if CUDA unavailable)

---

*Testing analysis: 2026-04-07*
