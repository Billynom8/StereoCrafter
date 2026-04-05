import os
import logging
import torch
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image

from core.common.video_io import read_video_frames
from core.ui.preview_buffer import PreviewFrameBuffer
from core.splatting.preview_rendering import PreviewRenderer

logger = logging.getLogger(__name__)


class PreviewController:
    """
    Headless controller for managing the Splatting Preview state.
    Handles video loading, frame extraction, and rendering coordination.
    """

    def __init__(self):
        self.renderer = PreviewRenderer()
        self.buffer = PreviewFrameBuffer(max_frames=500)

        # Internal State
        self.video_list: List[Dict[str, str]] = []
        self.current_video_index: int = -1
        self.source_reader = None
        self.depth_reader = None
        self.total_frames = 0
        self.fps = 0.0

    def load_video_list(self, source_path: str, depth_path: str, multi_map: bool = False):
        """Scans for matching videos and returns the list."""
        self.video_list = self.renderer.find_preview_sources(source_path, depth_path, multi_map)
        return self.video_list

    def set_current_video(self, index: int, params: Dict[str, Any]):
        """Opens the video readers for a specific index."""
        if not (0 <= index < len(self.video_list)):
            return False

        self.current_video_index = index
        entry = self.video_list[index]

        # Close old readers if they exist
        if hasattr(self.source_reader, "close"):
            self.source_reader.close()
        if hasattr(self.depth_reader, "close"):
            self.depth_reader.close()
        self.buffer.clear()

        # Use the robust video_io loader to get our readers
        # This handles the 10-bit depth and color-accuracy logic automatically
        source_data = read_video_frames(
            entry["source_video"],
            process_length=-1,
            set_pre_res=False,
            pre_res_width=0,
            pre_res_height=0,
            strict_ffmpeg_decode=params.get("strict_ffmpeg_decode", False),
        )

        depth_data = read_video_frames(
            entry["depth_map"], process_length=-1, set_pre_res=False, pre_res_width=0, pre_res_height=0, is_depth=True
        )

        self.source_reader = source_data[0]
        self.depth_reader = depth_data[0]
        self.fps = source_data[1]
        self.total_frames = source_data[7]  # total_to_process from return tuple

        return True

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
            # We use get_batch([idx]) to remain compatible with pipe readers
            src_batch = self.source_reader.get_batch([frame_idx]).asnumpy()
            depth_batch = self.depth_reader.get_batch([frame_idx]).asnumpy()

            # Convert to Tensors [1, C, H, W] for the Engine
            # Note: handle depth normalize if it's high-bit
            src_tensor = torch.from_numpy(src_batch).permute(0, 3, 1, 2).float() / 255.0
            depth_tensor = torch.from_numpy(depth_batch.astype("float32")).permute(0, 3, 1, 2)

            # If standard 8-bit depth, normalize
            if depth_batch.dtype == "uint8":
                depth_tensor /= 255.0
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
