"""
QThread Worker for managing playback timing.
Emits frame indices at the appropriate rate for smooth playback.
"""

import logging
from PySide6 import QtCore

logger = logging.getLogger(__name__)


class PlaybackWorker(QtCore.QObject):
    """
    Worker that manages playback timing and frame advancement.

    This worker does NOT render frames - it only tracks playback state
    and emits frame indices. The GUI connects to frame_advanced signal
    to trigger rendering.

    Usage:
        worker = PlaybackWorker(total_frames)
        worker.frame_advanced.connect(self.on_frame_advanced)
        worker.playback_finished.connect(self.on_playback_finished)

        # Start playback:
        worker.start()

        # Stop playback:
        worker.stop()

        # Toggle loop:
        worker.set_loop_enabled(True)
    """

    frame_advanced = QtCore.Signal(int)  # new frame index
    playback_finished = QtCore.Signal()
    playback_started = QtCore.Signal()
    playback_stopped = QtCore.Signal()

    def __init__(self, total_frames: int = 0, parent=None):
        super().__init__(parent)
        self._total_frames = total_frames
        self._current_frame = 0
        self._step = 1
        self._loop_enabled = False
        self._is_playing = False

    def set_total_frames(self, total: int):
        """Update total frame count (call when video changes)."""
        self._total_frames = total
        self._current_frame = min(self._current_frame, max(0, total - 1))

    def set_frame(self, frame_idx: int):
        """Set current frame index."""
        self._current_frame = max(0, min(frame_idx, self._total_frames - 1)) if self._total_frames > 0 else 0

    def get_frame(self) -> int:
        """Get current frame index."""
        return self._current_frame

    @QtCore.Slot()
    def start(self):
        """Start playback."""
        if self._total_frames <= 0:
            return
        self._is_playing = True
        self.playback_started.emit()

    @QtCore.Slot()
    def stop(self):
        """Stop playback."""
        self._is_playing = False
        self.playback_stopped.emit()

    @QtCore.Slot()
    def toggle(self):
        """Toggle playback state."""
        if self._is_playing:
            self.stop()
        else:
            self.start()

    def is_playing(self) -> bool:
        """Check if playback is active."""
        return self._is_playing

    def set_step(self, step: int):
        """Set frame step (1=normal, N=fast-forward)."""
        self._step = max(1, step)

    def set_loop_enabled(self, enabled: bool):
        """Enable or disable loop playback."""
        self._loop_enabled = enabled

    def is_loop_enabled(self) -> bool:
        """Check if loop is enabled."""
        return self._loop_enabled

    @QtCore.Slot()
    def tick(self):
        """
        Advance to next frame. Call this from a QTimer in the GUI.

        Emits:
            frame_advanced: With new frame index
            playback_finished: If playback ended (non-loop)
        """
        if not self._is_playing or self._total_frames <= 0:
            return

        next_frame = self._current_frame + self._step

        if next_frame >= self._total_frames:
            if self._loop_enabled and self._total_frames > 1:
                next_frame = 0
            else:
                self._is_playing = False
                self.playback_finished.emit()
                return

        self._current_frame = next_frame
        self.frame_advanced.emit(self._current_frame)
