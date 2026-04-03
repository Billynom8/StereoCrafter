"""High-level Orchestration Controller for Splatting.

Bridges the gap between UI (Tkinter/PyQt) or CLI and the core processing logic.
Holds the persistent state of a 'project' and manages batch lifecycles.
"""

import logging
import os
import queue
import threading
from typing import Dict, List, Optional, Any

from .batch_processing import BatchProcessor, ProcessingSettings, BatchSetupResult
from core.common.sidecar_manager import SidecarConfigManager

logger = logging.getLogger(__name__)


class SplattingController:
    """Orchestrates splatting operations, managing state and batch lifecycles."""

    def __init__(self, sidecar_manager: Optional[SidecarConfigManager] = None):
        self.progress_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.sidecar_manager = sidecar_manager or SidecarConfigManager()
        self.batch_processor = BatchProcessor(
            progress_queue=self.progress_queue, stop_event=self.stop_event, sidecar_manager=self.sidecar_manager
        )

        # Persistent state
        self.input_source_clips = ""
        self.input_depth_maps = ""
        self.output_splatted = ""
        self.video_list = []  # List of {"source_video": path, "depth_map": path}

        self.processing_thread = None

    def find_matching_pairs(self, source_path: str, depth_path: str, multi_map: bool = False) -> List[Dict[str, str]]:
        """
        Scans directories for matching source/depth video pairs.
        Replaces the logic previously in splatting_gui.py.
        """
        if not source_path or not depth_path:
            return []

        # Update internal state
        self.input_source_clips = source_path
        self.input_depth_maps = depth_path

        is_source_file = os.path.isfile(source_path)
        is_depth_file = os.path.isfile(depth_path)

        if is_source_file and is_depth_file:
            self.video_list = [{"source_video": source_path, "depth_map": depth_path}]
            return self.video_list

        if not os.path.isdir(source_path) or not os.path.isdir(depth_path):
            return []

        # Collect all source videos
        video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
        source_videos = []
        for filename in os.listdir(source_path):
            if filename.lower().endswith(video_extensions):
                source_videos.append(os.path.join(source_path, filename))

        if not source_videos:
            return []

        pairs = []

        # Multi-Map Scanning Logic
        if multi_map:
            depth_candidate_folders = []
            try:
                for entry in os.listdir(depth_path):
                    full_sub = os.path.join(depth_path, entry)
                    if os.path.isdir(full_sub) and entry.lower() != "sidecars":
                        depth_candidate_folders.append(full_sub)
            except Exception:
                pass

            for v_path in sorted(source_videos):
                base_name = os.path.splitext(os.path.basename(v_path))[0]
                matched = False
                for d_folder in depth_candidate_folders:
                    for ext in [".mp4", ".npz"]:
                        d_path = os.path.join(d_folder, f"{base_name}_depth{ext}")
                        if os.path.exists(d_path):
                            pairs.append({"source_video": v_path, "depth_map": d_path})
                            matched = True
                            break
                    if matched:
                        break
        else:
            # Normal Scanning Logic
            for v_path in sorted(source_videos):
                base_name = os.path.splitext(os.path.basename(v_path))[0]
                candidates = [
                    os.path.join(depth_path, f"{base_name}_depth.mp4"),
                    os.path.join(depth_path, f"{base_name}_depth.npz"),
                    os.path.join(depth_path, f"{base_name}.mp4"),
                    os.path.join(depth_path, f"{base_name}.npz"),
                ]
                for d_path in candidates:
                    if os.path.exists(d_path):
                        pairs.append({"source_video": v_path, "depth_map": d_path})
                        break

        self.video_list = pairs
        return pairs

    def start_batch(self, settings: ProcessingSettings, from_index: int = 0, to_index: Optional[int] = None):
        """Starts batch processing in a background thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Batch processing already in progress.")
            return

        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self.batch_processor.run_batch_process,
            kwargs={
                "settings": settings,
                "from_index": from_index,
                "to_index": to_index,
                "video_list": self.video_list,
            },
            daemon=True,
        )
        self.processing_thread.start()

    def start_auto_pass(self, settings: ProcessingSettings, from_index: int = 0, to_index: Optional[int] = None):
        """Starts AUTO-PASS (Pre-pass) in a background thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Processing already in progress.")
            return

        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self.batch_processor.run_auto_pass,
            kwargs={
                "settings": settings,
                "from_index": from_index,
                "to_index": to_index,
                "video_list": self.video_list,
            },
            daemon=True,
        )
        self.processing_thread.start()

    def stop(self):
        """Signals processing to stop."""
        self.stop_event.set()

    def get_progress(self) -> Optional[Any]:
        """Non-blocking poll of the progress queue."""
        try:
            return self.progress_queue.get_nowait()
        except queue.Empty:
            return None
