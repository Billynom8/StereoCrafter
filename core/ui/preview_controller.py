import os
import subprocess
import logging
import numpy as np
import torch
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image

from core.common.video_io import read_video_frames, get_video_stream_info
from core.ui.preview_buffer import PreviewFrameBuffer
from core.splatting.preview_rendering import PreviewRenderer
from core.common.sidecar_manager import SidecarConfigManager

logger = logging.getLogger(__name__)


class PreviewController:
    """
    Headless controller for managing the Splatting Preview state.
    Handles video loading, frame extraction, rendering coordination,
    playback state, depth detection, and sidecar persistence.
    """

    def __init__(self, sidecar_manager: Optional[SidecarConfigManager] = None):
        self.renderer = PreviewRenderer()
        self.buffer = PreviewFrameBuffer(max_frames=500)
        self.sidecar_manager = sidecar_manager or SidecarConfigManager()

        # Video State
        self.video_list: List[Dict[str, str]] = []
        self.current_video_index: int = -1
        self.source_reader = None
        self.depth_reader = None
        self.total_frames = 0
        self.fps = 0.0

        # Playback State
        self._is_playing: bool = False
        self._play_step: int = 1
        self._loop_enabled: bool = False
        self._current_frame: int = 0

        # Depth Bit-Depth Detection
        self._depth_path: Optional[str] = None
        self._depth_bit_depth: int = 8
        self._depth_is_high_bit: bool = False
        self._depth_native_w: Optional[int] = None
        self._depth_native_h: Optional[int] = None

        # Sidecar Configuration
        self._sidecar_folder: str = "./Sidecar"
        self._sidecar_extension: str = ".json"

    # =========================================================================
    # VIDEO LOADING & MANAGEMENT
    # =========================================================================

    def load_video_list(self, source_path: str, depth_path: str, multi_map: bool = False) -> List[Dict[str, str]]:
        """Scans for matching videos and returns the list."""
        self.video_list = self.renderer.find_preview_sources(source_path, depth_path, multi_map)
        return self.video_list

    def set_current_video(self, index: int, params: Dict[str, Any]) -> bool:
        """Opens the video readers for a specific index."""
        if not (0 <= index < len(self.video_list)):
            return False

        self.current_video_index = index
        entry = self.video_list[index]

        self._close_readers()
        self.buffer.clear()

        try:
            source_data = read_video_frames(
                entry["source_video"],
                process_length=-1,
                set_pre_res=False,
                pre_res_width=0,
                pre_res_height=0,
                strict_ffmpeg_decode=params.get("strict_ffmpeg_decode", False),
            )

            depth_data = read_video_frames(
                entry["depth_map"],
                process_length=-1,
                set_pre_res=False,
                pre_res_width=0,
                pre_res_height=0,
                is_depth=True,
            )

            self.source_reader = source_data[0]
            self.depth_reader = depth_data[0]
            self.fps = source_data[1]
            self.total_frames = source_data[7]

            # Probe depth bit-depth
            self._probe_depth_properties(entry["depth_map"], params)

            # Reset playback state
            self._current_frame = 0
            self._is_playing = False

            return True

        except Exception as e:
            logger.error(f"Failed to set video {index}: {e}")
            return False

    def _close_readers(self):
        """Close all video readers."""
        if self.source_reader and hasattr(self.source_reader, "close"):
            try:
                self.source_reader.close()
            except Exception:
                pass
        if self.depth_reader and hasattr(self.depth_reader, "close"):
            try:
                self.depth_reader.close()
            except Exception:
                pass
        self.source_reader = None
        self.depth_reader = None

    def get_current_video_entry(self) -> Optional[Dict[str, str]]:
        """Returns the current video entry dict or None."""
        if 0 <= self.current_video_index < len(self.video_list):
            return self.video_list[self.current_video_index]
        return None

    def get_total_frames(self) -> int:
        """Returns total frames for current video."""
        return self.total_frames

    def get_fps(self) -> float:
        """Returns FPS for current video."""
        return self.fps

    def get_video_dimensions(self) -> Tuple[int, int]:
        """Returns (width, height) for current source video."""
        if self.source_reader:
            try:
                frame = self.source_reader[0]
                return frame.shape[1], frame.shape[0]  # (width, height)
            except Exception:
                pass
        return (0, 0)

    # =========================================================================
    # PLAYBACK STATE MACHINE
    # =========================================================================

    def is_playing(self) -> bool:
        """Returns whether playback is active."""
        return self._is_playing

    def start_playback(self, step: int = 1):
        """Start playback with given frame step (1=normal, N=fast-forward)."""
        self._is_playing = True
        self._play_step = max(1, step)

    def stop_playback(self):
        """Stop playback."""
        self._is_playing = False

    def toggle_playback(self) -> bool:
        """Toggle playback state. Returns new state."""
        self._is_playing = not self._is_playing
        return self._is_playing

    def set_loop_enabled(self, enabled: bool):
        """Enable or disable loop playback."""
        self._loop_enabled = enabled

    def is_loop_enabled(self) -> bool:
        """Returns whether loop is enabled."""
        return self._loop_enabled

    def advance_frame(self) -> int:
        """
        Advance to next frame during playback.
        Returns the new frame index, or -1 if playback should stop.
        """
        if not self._is_playing or self.total_frames <= 0:
            return -1

        next_frame = self._current_frame + self._play_step

        if next_frame >= self.total_frames:
            if self._loop_enabled:
                next_frame = 0
            else:
                self._is_playing = False
                return -1

        self._current_frame = next_frame
        return self._current_frame

    def set_frame(self, frame_idx: int):
        """Set current frame index."""
        self._current_frame = max(0, min(frame_idx, self.total_frames - 1)) if self.total_frames > 0 else 0

    def get_frame_index(self) -> int:
        """Get current frame index."""
        return self._current_frame

    # =========================================================================
    # DEPTH BIT-DEPTH DETECTION
    # =========================================================================

    def _probe_depth_properties(self, depth_path: str, params: Dict[str, Any]):
        """Probe depth map for bit-depth and native dimensions."""
        self._depth_path = depth_path
        self._depth_bit_depth = 8
        self._depth_is_high_bit = False
        self._depth_native_w = None
        self._depth_native_h = None

        if not depth_path or not os.path.exists(depth_path):
            return

        try:
            depth_info = get_video_stream_info(depth_path)
            if not depth_info:
                return

            pix = str(depth_info.get("pix_fmt", "")).lower()
            profile = str(depth_info.get("profile", "")).lower()

            # Infer bit depth from pixel format
            if "p16" in pix or "16" in pix or pix.startswith("gray16"):
                self._depth_bit_depth = 16
            elif "p12" in pix or "12" in pix:
                self._depth_bit_depth = 12
            elif "p10" in pix or "10" in pix or "main10" in profile:
                self._depth_bit_depth = 10
            else:
                self._depth_bit_depth = 8

            self._depth_is_high_bit = self._depth_bit_depth > 8

            # Get native dimensions from metadata
            self._depth_native_w = int(depth_info.get("width", 0))
            self._depth_native_h = int(depth_info.get("height", 0))

            logger.debug(
                f"Depth probed: {self._depth_bit_depth}-bit, native {self._depth_native_w}x{self._depth_native_h}"
            )

        except Exception as e:
            logger.warning(f"Depth probe failed for '{depth_path}': {e}")

    def get_depth_bit_depth(self) -> int:
        """Returns detected depth bit-depth (8, 10, 12, or 16)."""
        return self._depth_bit_depth

    def is_depth_high_bit(self) -> bool:
        """Returns True if depth is >8-bit."""
        return self._depth_is_high_bit

    # TODO: Remove this method - it's not needed anymore
    def _read_depth_frame_ffmpeg(self, frame_idx: int) -> np.ndarray:
        """
        Decode a single depth frame preserving 10-bit+ using FFmpeg.
        Returns raw uint16 numpy array.
        """
        if not self._depth_path or not self._depth_native_w or not self._depth_native_h:
            raise RuntimeError("Depth path/size not initialized for FFmpeg decode")

        # Calculate timestamp
        fps = self.fps if self.fps > 0 else 30.0
        t = float(frame_idx) / fps
        w, h = int(self._depth_native_w), int(self._depth_native_h)
        expected_bytes = w * h * 2  # gray16le

        def run_ffmpeg(vf_filter: str) -> bytes:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{t:.6f}",
                "-i",
                self._depth_path,
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "1",
                "-vf",
                vf_filter,
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out = proc.stdout.read(expected_bytes) if proc.stdout else b""
            finally:
                proc.terminate()
            return out

        # Try luma plane extraction first (for YUV-encoded depth)
        buf = run_ffmpeg("extractplanes=y,format=gray16le")
        if len(buf) != expected_bytes:
            buf = run_ffmpeg("format=gray16le")

        if len(buf) != expected_bytes:
            raise RuntimeError(f"FFmpeg depth decode got {len(buf)} bytes, expected {expected_bytes}")

        return np.frombuffer(buf, dtype=np.uint16).reshape(h, w)

    # =========================================================================
    # FRAME RENDERING
    # =========================================================================

    def get_frame(self, frame_idx: int, params: Dict[str, Any]) -> Optional[Image.Image]:
        """Fetches and renders a specific frame index based on the provided params."""
        if self.source_reader is None:
            return None

        # 1. Handle complexity: Border Percentages (Ported from splatting_gui.py)
        # This converts the "Mode" (Manual/Auto/Adv) into raw percentages for the engine
        processed_params = self._calculate_border_percentages(params)

        # 2. Check Cache
        video_path = self.video_list[self.current_video_index]["source_video"]
        if self.buffer.check_and_update_buffer(processed_params, video_path):
            logger.debug("Buffer invalidated by parameter change.")

        cached = self.buffer.get_cached_frame(frame_idx)
        if cached:
            return cached

        # 3. Pull raw frames from readers
        try:
            src_batch = self.source_reader.get_batch([frame_idx]).asnumpy()

            # Handle high-bit-depth depth maps via FFmpeg
            if self._depth_is_high_bit and self._depth_path:
                # FFmpegDepthPipeReader returns (1, H, W, 1) uint16
                depth_batch = self.depth_reader.get_batch([frame_idx]).asnumpy()
                logger.debug(f"High-bit depth batch shape: {depth_batch.shape}, dtype: {depth_batch.dtype}")
                # Normalize based on bit depth
                max_val = (1 << self._depth_bit_depth) - 1
                depth_batch = depth_batch.astype("float32") / max_val
            else:
                depth_batch = self.depth_reader.get_batch([frame_idx]).asnumpy()
                logger.debug(f"8-bit depth batch shape: {depth_batch.shape}, dtype: {depth_batch.dtype}")

            logger.debug(f"Source batch shape: {src_batch.shape}")
            src_tensor = torch.from_numpy(src_batch).permute(0, 3, 1, 2).float() / 255.0
            depth_tensor = torch.from_numpy(depth_batch.astype("float32")).permute(0, 3, 1, 2)
            logger.debug(f"Source tensor shape: {src_tensor.shape}, Depth tensor shape: {depth_tensor.shape}")

            if depth_batch.dtype == "uint8":
                depth_tensor = depth_tensor / 255.0

        except Exception as e:
            logger.error(f"Failed to read frame {frame_idx}: {e}")
            return None

        # 4. Render
        mode = processed_params.get("preview_source", "Splat Result")
        rendered_image = self.renderer.render_preview_frame(src_tensor, depth_tensor, processed_params, mode)

        # 5. Cache and return
        if rendered_image:
            self.buffer.cache_frame(frame_idx, rendered_image)

        return rendered_image

    def _calculate_border_percentages(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Logic extracted from splatting_gui.py to resolve border settings."""
        p = params.copy()
        mode = p.get("border_mode", "Off")
        l_pct, r_pct = 0.0, 0.0

        if mode == "Auto Basic":
            w = float(p.get("border_width", 0.0))
            l_pct, r_pct = w, w
        elif mode == "Auto Adv.":
            l_pct = float(p.get("auto_border_L", 0.0))
            r_pct = float(p.get("auto_border_R", 0.0))
        elif mode == "Manual":
            w = float(p.get("border_width", 0.0))
            b = float(p.get("border_bias", 0.0))
            if b <= 0:
                l_pct = w
                r_pct = w * (1.0 + b)
            else:
                r_pct = w
                l_pct = w * (1.0 - b)

        p["left_border_pct"] = l_pct
        p["right_border_pct"] = r_pct
        return p

    # =========================================================================
    # SIDECAR MANAGEMENT
    # =========================================================================

    def set_sidecar_folder(self, folder: str):
        """Set the folder where sidecar files are stored."""
        self._sidecar_folder = folder

    def get_sidecar_path(self) -> Optional[str]:
        """Build the path to the sidecar file for the current video."""
        entry = self.get_current_video_entry()
        if not entry:
            return None

        filename = os.path.basename(entry.get("source_video", ""))
        if not filename:
            return None

        sidecar_filename = os.path.splitext(filename)[0] + self._sidecar_extension

        if not os.path.exists(self._sidecar_folder):
            os.makedirs(self._sidecar_folder, exist_ok=True)

        return os.path.join(self._sidecar_folder, sidecar_filename)

    def load_sidecar(self) -> Dict[str, Any]:
        """Load sidecar data for the current video using SidecarConfigManager."""
        sidecar_path = self.get_sidecar_path()
        if not sidecar_path or not os.path.exists(sidecar_path):
            return self.sidecar_manager._get_defaults()

        return self.sidecar_manager.load_sidecar_data(sidecar_path)

    def save_sidecar(self, params: Dict[str, Any]) -> bool:
        """Save current params to sidecar using SidecarConfigManager schema."""
        sidecar_path = self.get_sidecar_path()
        if not sidecar_path:
            return False

        # Map GUI keys to sidecar keys
        sidecar_data = self._map_params_to_sidecar(params)
        return self.sidecar_manager.save_sidecar_data(sidecar_path, sidecar_data)

    def _map_params_to_sidecar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map GUI parameter keys to sidecar schema keys."""
        mapping = {
            "max_disp": "max_disparity",
            "convergence_point": "convergence_plane",
            "gamma": "gamma",
            "dilate_x": "depth_dilate_size_x",
            "dilate_y": "depth_dilate_size_y",
            "blur_x": "depth_blur_size_x",
            "blur_y": "depth_blur_size_y",
            "border_mode": "border_mode",
            "border_width": "left_border",
            "view_bias": "input_bias",
            "flip_horizontal": "flip_horizontal",
            "preview_source": "selected_depth_map",
        }

        sidecar_data = {}
        for gui_key, sidecar_key in mapping.items():
            if gui_key in params:
                sidecar_data[sidecar_key] = params[gui_key]

        return sidecar_data

    def _map_sidecar_to_params(self, sidecar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map sidecar schema keys to GUI parameter keys."""
        reverse_mapping = {
            "max_disparity": "max_disp",
            "convergence_plane": "convergence_point",
            "gamma": "gamma",
            "depth_dilate_size_x": "dilate_x",
            "depth_dilate_size_y": "dilate_y",
            "depth_blur_size_x": "blur_x",
            "depth_blur_size_y": "blur_y",
            "border_mode": "border_mode",
            "left_border": "border_width",
            "input_bias": "view_bias",
            "flip_horizontal": "flip_horizontal",
            "selected_depth_map": "preview_source",
        }

        params = {}
        for sidecar_key, gui_key in reverse_mapping.items():
            if sidecar_key in sidecar_data:
                params[gui_key] = sidecar_data[sidecar_key]

        return params

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup(self):
        """Release all resources."""
        self._close_readers()
        self.buffer.clear()
        self._is_playing = False
