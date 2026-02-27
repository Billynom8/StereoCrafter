# Splatting GUI Changelog

All notable changes to the splatting GUI and related components.

## Version 26-02-27.4

### Added

- **10-bit & Color Tagging**: Added support for 10-bit HEVC (libx265/hevc_nvenc) and DNxHR HQX encoding with accurate color space metadata (BT.2020 PQ/HLG).
- **DNxHR Split Mode**: New output mode for high-resolution dual-stream delivery ([Occlusion Mask] and [Right Eye] as separate files).
- **SBS Preview Window**: Integrated real-time Side-By-Side (SBS) preview toggle for immediate stereo alignment checks.
- **Strict FFmpeg Decode**: New toggle to force bit-accurate depth map reading via FFmpeg pipes, bypassing 8-bit truncation issues in some decoders.
- **Diagnostic Capture Suite**: Integrated PNG export for "Map Test" and "Splat Test" frames, featuring auto-switching preview sources and metadata labeling.

### Fixed

- **Aspect Ratio Parity**: Fixed horizontal stretching mismatch between preview and render; depth maps are now resized early to match source video aspect ratios.
- **Diagnostic Quad-Cropping**: Fixed quadrant cropping logic to correctly handle both 2-panel (Dual) and 4-panel grid layouts in PNG exports.
- **Depth Unification**: Disabled automatic TV-range expansion in preview to ensure visual parity with "raw" render outputs.
- **Bit-Depth Detection**: Unified bit-depth inference between preview and render paths to prevent contrast mismatches during normalization.

### Changed

- **Renderer Optimization**: Diagnostic captures now force a 4-panel grid internally to ensure all data (Depth, Occlusion, etc.) is available regardless of output mode.

## Version 26-02-26.2

### Bug Fixes (v26-02-26.2)

- **Depth Preprocessing**: Fixed issue where preview depth pre-processing was using local and not the core library functions.
- **Depth Normalization**: unified and refactored depth normalization logic between preview and batch processing.

## Version 26-02-21.0

### Bug Fixes (v26-02-21.0)

- **10-bit Precision Loss**: Restored `float32` rendering pipeline to prevent bit-depth truncation to 8-bit before encoding.
- **HDR/HLG Output**: Fixed issue where BT.2020 PQ and HLG selections incorrectly produced BT.709 output.
- **Missing GUI Options**: Restored "BT.709 L/F" and "BT.2020 PQ/HLG" modes to the Color Tags dropdown.

### Improvements (v26-02-21.0)

- **HDR Detection**: Enhanced `stereocrafter_util.py` to correctly detect HLG transfer characteristics for 10-bit HEVC encoding.
- **Legacy Compatibility**: Added internal mappings for old "BT.709" and "BT.2020" config strings to ensure high-quality profile selection.

---

## Version 26-02-07.1

### Added (v26-02-07.1)

- **Manual Mode for Auto-Convergence:** New "Manual" mode writes current slider values to sidecars during AUTO-PASS without calculating auto-convergence.
- **Sidecar Migration Menu Items:** Two new File menu options to move sidecars between folders:
  - "Sidecars: Depth → Source (remove _depth)" - moves sidecars from depth folder to source folder
  - "Sidecars: Source → Depth (add _depth)" - moves sidecars from source folder to depth folder

### Changed (v26-02-07.1)

- **Auto-Convergence "Off" Behavior:** AUTO-PASS no longer overwrites sidecar values when set to "Off" - existing sidecar values are now preserved.
- **AUTO-PASS Border Mode:** When GUI Border Mode is "Auto Basic" or "Auto Adv.", AUTO-PASS now stores values in `auto_border_L`/`auto_border_R` fields (for UI caching) and keeps `border_mode` as "Auto Basic"/"Auto Adv." instead of switching to "Manual".

### Bug Fixes (v26-02-07.1)

- **Auto-Convergence Cache Clearing:** Fixed clip navigation not clearing cached Average/Peak values, which caused incorrect values to be applied when switching between clips.

---

## Version 2026-02-04

### Refactored (v2026-02-04)

- **Code Consolidation**: Removed duplicate depth processing code from `splatting_gui.py` (lines 7018-7234)
  - `compute_global_depth_stats()` function now imported from `core.splatting.depth_processing`
  - `load_pre_rendered_depth()` function now imported from `core.splatting.depth_processing`
  - Removed local redefinitions of `_NumpyBatch` and `_ResizingDepthReader` classes
  - Eliminated ~216 lines of duplicated code

### Changed (v2026-02-04)

- **Slider Behavior Enhancement** (`dependency/stereocrafter_util.py`):
  - **Middle-click**: Slider now jumps directly to mouse pointer position (matches frame scrubber behavior)
  - **Right-click**: Resets slider to default value
  - **Left-click**: Maintains original stepped increment behavior
- **Browes Folder Buttons:** Right-Click opens explorer window at paths location.
- **Combined Scanning:** `Auto Convergence` and `Boarder Depth` into a single pass.

### Bug Fixes (v2026-02-04)

- Slider middle-click functionality now provides precise positioning like the frame scrubber
