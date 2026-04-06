"""
QThread Worker for rendering preview frames in a background thread.
Keeps the GUI responsive during heavy rendering operations.
"""

import logging
from typing import Dict, Any
from PySide6 import QtCore

logger = logging.getLogger(__name__)


class PreviewRenderWorker(QtCore.QObject):
    """
    Worker that renders preview frames on a background thread.

    Usage:
        worker = PreviewRenderWorker(controller)
        worker.frame_ready.connect(self.on_frame_ready)
        worker.error.connect(self.on_error)

        # In a QThread:
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.start()

        # Trigger render:
        QtCore.QMetaObject.invokeMethod(worker, "render_frame",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, frame_idx),
            QtCore.Q_ARG(dict, params))
    """

    frame_ready = QtCore.Signal(object)  # PIL Image
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._is_running = True

    @QtCore.Slot()
    def stop(self):
        """Stop accepting new render requests."""
        self._is_running = False

    @QtCore.Slot(int, dict)
    def render_frame(self, frame_idx: int, params: Dict[str, Any]):
        """
        Render a single frame. Emits frame_ready or error signal.

        Args:
            frame_idx: Frame index to render
            params: Rendering parameters dict
        """
        if not self._is_running:
            return

        try:
            img = self.controller.get_frame(frame_idx, params)
            if img is not None:
                self.frame_ready.emit(img)
            else:
                self.error.emit(f"No image returned for frame {frame_idx}")
        except Exception as e:
            logger.exception(f"Render worker error: {e}")
            self.error.emit(str(e))

    @QtCore.Slot()
    def cleanup(self):
        """Cleanup when thread is shutting down."""
        self._is_running = False
        self.finished.emit()
