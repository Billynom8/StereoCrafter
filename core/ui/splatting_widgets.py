import logging
import os
from PySide6 import QtWidgets, QtCore
from PySide6.QtGui import QDoubleValidator, QIntValidator

logger = logging.getLogger(__name__)


class DropFilter(QtCore.QObject):
    """Event filter that enables drag-and-drop on any widget."""

    files_dropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        elif event.type() == QtCore.QEvent.Type.DragLeave:
            return True
        elif event.type() == QtCore.QEvent.Type.Drop:
            urls = event.mimeData().urls()
            paths = [url.toLocalFile() for url in urls]
            if paths:
                self.files_dropped.emit(paths)
            return True
        return super().eventFilter(obj, event)


class InputOutputWidget(QtWidgets.QWidget):
    depth_file_dropped = QtCore.Signal(str)  # Signal for depth file drops to bypass suffix filter

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._drop_filters = {}  # Store filters to prevent garbage collection
        self._setup_validators()
        self._enable_drag_drop()
        self._connect_signals()

    def _setup_validators(self):
        self.ui.lineEdit_mesh_extrusion.setValidator(QDoubleValidator(-1.0, 5.0, 3, self))
        self.ui.lineEdit_mesh_density.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        self.ui.lineEdit_mesh_bias.setValidator(QDoubleValidator(-1.0, 1.0, 3, self))
        self.ui.lineEdit_mesh_dolly.setValidator(QDoubleValidator(0.0, 100.0, 3, self))

    def _install_drop_filter(self, widget, callback):
        """Install a drop event filter on a widget."""
        widget.setAcceptDrops(True)
        drop_filter = DropFilter(widget)
        drop_filter.files_dropped.connect(callback)
        widget.installEventFilter(drop_filter)
        self._drop_filters[widget] = drop_filter
        return drop_filter

    def _enable_drag_drop(self):
        """Enable drag-drop on line edits and buttons."""
        # Enable on line edits
        self._install_drop_filter(self.ui.lineEdit_input_source, self._on_source_dropped)
        self._install_drop_filter(self.ui.lineEdit_input_depth, self._on_depth_dropped)
        self._install_drop_filter(self.ui.lineEdit_output_splatted, self._on_output_dropped)
        self._install_drop_filter(self.ui.lineEdit_inout_sidecar, self._on_sidecar_dropped)

        # Enable on browse buttons
        self._install_drop_filter(self.ui.pushButton_browse_source, self._on_source_dropped)
        self._install_drop_filter(self.ui.pushButton_select_source, self._on_source_dropped)
        self._install_drop_filter(self.ui.pushButton_browse_depth, self._on_depth_dropped)
        self._install_drop_filter(self.ui.pushButton_select_depth, self._on_depth_dropped)
        self._install_drop_filter(self.ui.pushButton_browse_output, self._on_output_dropped)
        self._install_drop_filter(self.ui.pushButton_browse_sidecar, self._on_sidecar_dropped)

    def _on_source_dropped(self, paths):
        """Handle dropped source path(s)."""
        if paths:
            path = paths[0]
            if os.path.isfile(path):
                path = os.path.dirname(path)
            self.ui.lineEdit_input_source.setText(path)
            logger.info(f"Source path set via drop: {path}")

    def _on_depth_dropped(self, paths):
        """Handle dropped depth path(s). Emit signal to bypass suffix filter if file."""
        if paths:
            path = paths[0]
            if os.path.isfile(path):
                # Emit signal for file drop to bypass suffix filter
                self.depth_file_dropped.emit(path)
                self.ui.lineEdit_input_depth.setText(path)
                logger.info(f"Depth file set via drop (bypasses suffix filter): {path}")
            else:
                self.ui.lineEdit_input_depth.setText(path)
                logger.info(f"Depth folder set via drop: {path}")

    def _on_output_dropped(self, paths):
        """Handle dropped output path(s). Extract directory if file."""
        if paths:
            path = paths[0]
            if os.path.isfile(path):
                path = os.path.dirname(path)
            self.ui.lineEdit_output_splatted.setText(path)
            logger.info(f"Output path set via drop: {path}")

    def _on_sidecar_dropped(self, paths):
        """Handle dropped sidecar path(s). Extract directory if file."""
        if paths:
            path = paths[0]
            if os.path.isfile(path):
                path = os.path.dirname(path)
            self.ui.lineEdit_inout_sidecar.setText(path)
            logger.info(f"Sidecar path set via drop: {path}")

    def _connect_signals(self):
        # Browse buttons - use current path as starting directory
        self.ui.pushButton_browse_source.clicked.connect(self._browse_source)
        self.ui.pushButton_browse_depth.clicked.connect(self._browse_depth)
        self.ui.pushButton_browse_output.clicked.connect(self._browse_output)
        self.ui.pushButton_browse_sidecar.clicked.connect(self._browse_sidecar)

        # Select file buttons
        self.ui.pushButton_select_source.clicked.connect(self._select_source_file)
        self.ui.pushButton_select_depth.clicked.connect(self._select_depth_file)

    def _get_start_dir(self, line_edit):
        """Get starting directory from line edit, fallback to workspace if invalid."""
        path = line_edit.text()
        if path:
            if os.path.isfile(path):
                return os.path.dirname(path)
            if os.path.isdir(path):
                return path
        return "./workspace"

    def _browse_source(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Source Directory", self._get_start_dir(self.ui.lineEdit_input_source)
        )
        if folder:
            self.ui.lineEdit_input_source.setText(folder)

    def _browse_depth(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Depth Directory", self._get_start_dir(self.ui.lineEdit_input_depth)
        )
        if folder:
            self.ui.lineEdit_input_depth.setText(folder)

    def _browse_output(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._get_start_dir(self.ui.lineEdit_output_splatted)
        )
        if folder:
            self.ui.lineEdit_output_splatted.setText(folder)

    def _browse_sidecar(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Sidecar Directory", self._get_start_dir(self.ui.lineEdit_inout_sidecar)
        )
        if folder:
            self.ui.lineEdit_inout_sidecar.setText(folder)

    def _select_source_file(self):
        start_dir = self._get_start_dir(self.ui.lineEdit_input_source)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Source Video File", start_dir, "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*)"
        )
        if file_path:
            self.ui.lineEdit_input_source.setText(file_path)
            logger.info(f"Source file selected: {file_path}")

    def _select_depth_file(self):
        start_dir = self._get_start_dir(self.ui.lineEdit_input_depth)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Depth Map File", start_dir, "Depth Files (*_depth.* *.exr *.png);;All Files (*)"
        )
        if file_path:
            # Emit signal to bypass suffix filter
            self.depth_file_dropped.emit(file_path)
            self.ui.lineEdit_input_depth.setText(file_path)
            logger.info(f"Depth file selected (bypasses suffix filter): {file_path}")

    def get_source_path(self):
        return self.ui.lineEdit_input_source.text()

    def get_depth_path(self):
        return self.ui.lineEdit_input_depth.text()

    def get_output_path(self):
        return self.ui.lineEdit_output_splatted.text()

    def get_sidecar_path(self):
        return self.ui.lineEdit_inout_sidecar.text()

    def is_multi_map(self):
        return self.ui.checkBox_multi_map.isChecked()

    def set_source_path(self, path):
        self.ui.lineEdit_input_source.setText(path)

    def set_depth_path(self, path):
        self.ui.lineEdit_input_depth.setText(path)

    def set_output_path(self, path):
        self.ui.lineEdit_output_splatted.setText(path)

    def set_sidecar_path(self, path):
        self.ui.lineEdit_inout_sidecar.setText(path)

    def set_multi_map(self, enabled):
        self.ui.checkBox_multi_map.setChecked(enabled)


class PreviewControlsWidget(QtWidgets.QWidget):
    frame_changed = QtCore.Signal(int)
    video_changed = QtCore.Signal(int)
    preview_mode_changed = QtCore.Signal(str)
    playback_requested = QtCore.Signal()
    scale_changed = QtCore.Signal(str)

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.ui.comboBox_preview_source.clear()
        modes = [
            "Splat Result",
            "Splat Result(Low)",
            "Occlusion Mask",
            "Occlusion Mask(Low)",
            "Original (Left Eye)",
            "Depth Map",
            "Depth Map (Color)",
            "Anaglyph 3D",
            "Dubois Anaglyph",
            "Optimized Anaglyph",
            "Side-by-Side",
            "Wigglegram",
            "Mesh Warp",
            "SBS + Mesh",
        ]
        self.ui.comboBox_preview_source.addItems(modes)
        logger.info(f"Initialized Preview Source with {len(modes)} modes.")

        self.ui.comboBox_border.clear()
        self.ui.comboBox_border.addItems(["Off", "Auto Basic", "Auto Adv.", "Manual"])

        self.ui.comboBox_preview_scale.clear()
        percentage_values = ["Auto"] + [
            "250%",
            "240%",
            "230%",
            "220%",
            "210%",
            "200%",
            "190%",
            "180%",
            "170%",
            "160%",
            "150%",
            "145%",
            "140%",
            "135%",
            "130%",
            "125%",
            "120%",
            "115%",
            "110%",
            "105%",
            "100%",
            "95%",
            "90%",
            "85%",
            "80%",
            "75%",
            "70%",
            "65%",
            "60%",
            "55%",
            "50%",
            "25%",
        ]
        self.ui.comboBox_preview_scale.addItems(percentage_values)
        self.ui.comboBox_preview_scale.setCurrentText("Auto")

    def _connect_signals(self):
        self.ui.horizontalSlider.valueChanged.connect(self._on_frame_change)
        self.ui.pushButton_next.clicked.connect(lambda: self._on_video_change(1))
        self.ui.pushButton_prev.clicked.connect(lambda: self._on_video_change(-1))
        self.ui.pushButton_play.clicked.connect(self._on_playback)
        self.ui.pushButton_fast_forward.clicked.connect(lambda: self._on_playback(fast=True))
        self.ui.comboBox_preview_scale.currentTextChanged.connect(self._on_scale_change)

    def _on_frame_change(self, frame):
        self.frame_changed.emit(frame)

    def _on_video_change(self, delta):
        self.video_changed.emit(delta)

    def _on_playback(self, fast=False):
        self.playback_requested.emit()

    def _on_scale_change(self, value):
        self.scale_changed.emit(value)

    def init_ui_defaults(self):
        self.ui.lineEdit_jump_to.setValidator(QIntValidator(1, 999999, self))
        idx = self.ui.comboBox_preview_source.findText("Splat Result")
        if idx >= 0:
            self.ui.comboBox_preview_source.setCurrentIndex(idx)

    def get_frame(self):
        return self.ui.horizontalSlider.value()

    def set_frame(self, frame):
        self.ui.horizontalSlider.setValue(frame)

    def get_total_frames(self):
        return self.ui.horizontalSlider.maximum() + 1

    def set_total_frames(self, total):
        self.ui.horizontalSlider.blockSignals(True)
        self.ui.horizontalSlider.setMaximum(total - 1)
        self.ui.horizontalSlider.blockSignals(False)

    def get_preview_mode(self):
        return self.ui.comboBox_preview_source.currentText()

    def set_preview_mode(self, mode):
        idx = self.ui.comboBox_preview_source.findText(mode)
        if idx >= 0:
            self.ui.comboBox_preview_source.setCurrentIndex(idx)

    def get_scale(self):
        return self.ui.comboBox_preview_scale.currentText()

    def set_scale(self, scale):
        idx = self.ui.comboBox_preview_scale.findText(scale)
        if idx >= 0:
            self.ui.comboBox_preview_scale.setCurrentIndex(idx)

    def get_ff_speed(self):
        return self.ui.spinBox_ff_speed.value()

    def set_ff_speed(self, speed):
        self.ui.spinBox_ff_speed.setValue(speed)

    def get_current_video_idx(self):
        text = self.ui.lineEdit_jump_to.text()
        try:
            return max(0, int(text) - 1)
        except ValueError:
            return 0


class StereoProjectionWidget(QtWidgets.QWidget):
    value_changed = QtCore.Signal()

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._connect_signals()

    def _connect_signals(self):
        # Use sliderReleased to avoid lag during drag
        # Labels still update in real-time for visual feedback
        self.ui.horizontalSlider_disparity.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_convergence.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_gamma.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_border_width.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_border_bias.valueChanged.connect(self.update_labels)

        # Emit value_changed only on slider release to trigger re-render
        self.ui.horizontalSlider_disparity.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_convergence.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_gamma.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_border_width.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_border_bias.sliderReleased.connect(self._on_slider_release)

        # Cross view checkbox triggers immediate refresh
        self.ui.checkBox_cross_view.toggled.connect(self._on_slider_release)

    def _on_slider_release(self):
        """Emit value_changed when slider is released or checkbox toggled."""
        self.value_changed.emit()

    def init_ui_defaults(self):
        self.ui.horizontalSlider_border_bias.setMinimum(0)
        self.ui.horizontalSlider_border_bias.setMaximum(100)
        self.update_labels()

    def update_labels(self):
        self.ui.label_disparity_value.setText(str(self.ui.horizontalSlider_disparity.value()))
        self.ui.label_convergence_value.setText(f"{self.ui.horizontalSlider_convergence.value() / 100.0:.2f}")
        self.ui.label_gamma_value.setText(f"{self.ui.horizontalSlider_gamma.value() / 100.0:.2f}")
        self.ui.label_border_width_value.setText(str(self.ui.horizontalSlider_border_width.value()))
        bias_val = (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0
        self.ui.label_bias_value.setText(f"{bias_val:.2f}")

    def get_disparity(self):
        return self.ui.horizontalSlider_disparity.value()

    def get_convergence(self):
        return self.ui.horizontalSlider_convergence.value() / 100.0

    def get_gamma(self):
        return self.ui.horizontalSlider_gamma.value() / 100.0

    def get_border_mode(self):
        return self.ui.comboBox_border.currentText()

    def get_border_width(self):
        return self.ui.horizontalSlider_border_width.value()

    def get_border_bias(self):
        return (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0

    def is_cross_view(self):
        return self.ui.checkBox_cross_view.isChecked()

    def set_disparity(self, value):
        self.ui.horizontalSlider_disparity.setValue(value)

    def set_convergence(self, value):
        self.ui.horizontalSlider_convergence.setValue(int(value * 100))

    def set_gamma(self, value):
        self.ui.horizontalSlider_gamma.setValue(int(value * 100))

    def set_border_mode(self, mode):
        idx = self.ui.comboBox_border.findText(mode)
        if idx >= 0:
            self.ui.comboBox_border.setCurrentIndex(idx)

    def set_border_width(self, value):
        self.ui.horizontalSlider_border_width.setValue(value)

    def set_border_bias(self, value):
        self.ui.horizontalSlider_border_bias.setValue(int((value * 50) + 50))

    def set_cross_view(self, enabled):
        self.ui.checkBox_cross_view.setChecked(enabled)

    def is_resume_enabled(self):
        """Returns whether the Resume (move to finished) checkbox is checked."""
        return self.ui.checkBox_resume.isChecked()

    def set_resume_enabled(self, enabled):
        """Set the Resume (move to finished) checkbox state."""
        self.ui.checkBox_resume.setChecked(enabled)


class DepthPreprocessingWidget(QtWidgets.QWidget):
    value_changed = QtCore.Signal()

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._connect_signals()

    def _connect_signals(self):
        # Use sliderReleased for depth preprocessing to avoid lag during drag
        # Labels still update in real-time via valueChanged for visual feedback
        self.ui.horizontalSlider_dilate_x.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_dilate_y.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_blur_x.valueChanged.connect(self.update_labels)
        self.ui.horizontalSlider_blur_y.valueChanged.connect(self.update_labels)

        # Emit value_changed only on slider release to trigger re-render
        self.ui.horizontalSlider_dilate_x.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_dilate_y.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_blur_x.sliderReleased.connect(self._on_slider_release)
        self.ui.horizontalSlider_blur_y.sliderReleased.connect(self._on_slider_release)

        self.ui.lineEdit_blur_bias.editingFinished.connect(self._on_blur_bias_change)

    def _on_slider_release(self):
        """Emit value_changed when slider is released."""
        self.value_changed.emit()

    def _on_blur_bias_change(self):
        self.value_changed.emit()

    def update_labels(self):
        self.ui.label_dilate_x_value.setText(str(self.ui.horizontalSlider_dilate_x.value()))
        self.ui.label_dilate_y_value.setText(str(self.ui.horizontalSlider_dilate_y.value()))
        self.ui.label_blue_x_value.setText(str(self.ui.horizontalSlider_blur_x.value()))
        self.ui.label_blur_y_value.setText(str(self.ui.horizontalSlider_blur_y.value()))

    def get_dilate_x(self):
        return self.ui.horizontalSlider_dilate_x.value()

    def get_dilate_y(self):
        return self.ui.horizontalSlider_dilate_y.value() / 2.0

    def get_blur_x(self):
        return self.ui.horizontalSlider_blur_x.value()

    def get_blur_y(self):
        return self.ui.horizontalSlider_blur_y.value()

    def get_blur_bias(self):
        return float(self.ui.lineEdit_blur_bias.text() or 0.5)

    def set_dilate_x(self, value):
        self.ui.horizontalSlider_dilate_x.setValue(value)

    def set_dilate_y(self, value):
        self.ui.horizontalSlider_dilate_y.setValue(int(value * 2.0))

    def set_blur_x(self, value):
        self.ui.horizontalSlider_blur_x.setValue(value)

    def set_blur_y(self, value):
        self.ui.horizontalSlider_blur_y.setValue(value)

    def set_blur_bias(self, value):
        self.ui.lineEdit_blur_bias.setText(str(value))


class SplattingSettingsWidget(QtWidgets.QWidget):
    value_changed = QtCore.Signal()

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._init_defaults()
        self._connect_signals()

    def _init_defaults(self):
        self.ui.comboBox_mask_type.clear()
        self.ui.comboBox_mask_type.addItems(["SC", "M2S"])
        self.ui.comboBox_mask_type.setCurrentText("SC")

    def _connect_signals(self):
        self.ui.comboBox_mask_type.currentTextChanged.connect(self._on_change)
        self.ui.lineEdit_mesh_extrusion.editingFinished.connect(self._on_change)
        self.ui.lineEdit_mesh_density.editingFinished.connect(self._on_change)
        self.ui.lineEdit_mesh_bias.editingFinished.connect(self._on_change)
        self.ui.lineEdit_mesh_dolly.editingFinished.connect(self._on_change)

    def _on_change(self, value=None):
        self.value_changed.emit()

    def get_process_length(self):
        return int(self.ui.lineEdit_process_length.text() or 0)

    def get_mesh_extrusion(self):
        return float(self.ui.lineEdit_mesh_extrusion.text() or 0.5)

    def get_mesh_density(self):
        return float(self.ui.lineEdit_mesh_density.text() or 0.5)

    def get_mesh_bias(self):
        return float(self.ui.lineEdit_mesh_bias.text() or 0.0)

    def get_mesh_dolly(self):
        return float(self.ui.lineEdit_mesh_dolly.text() or 0.0)

    def get_mask_type(self):
        return self.ui.comboBox_mask_type.currentText() or "SC"

    def set_process_length(self, value):
        self.ui.lineEdit_process_length.setText(str(value))

    def set_mesh_extrusion(self, value):
        self.ui.lineEdit_mesh_extrusion.setText(str(value))

    def set_mesh_density(self, value):
        self.ui.lineEdit_mesh_density.setText(str(value))

    def set_mesh_bias(self, value):
        self.ui.lineEdit_mesh_bias.setText(str(value))

    def set_mesh_dolly(self, value):
        self.ui.lineEdit_mesh_dolly.setText(str(value))

    def set_mask_type(self, mask_type):
        idx = self.ui.comboBox_mask_type.findText(mask_type)
        if idx >= 0:
            self.ui.comboBox_mask_type.setCurrentIndex(idx)
        else:
            self.ui.comboBox_mask_type.setCurrentText(mask_type)


class ProcessingResolutionWidget(QtWidgets.QWidget):
    value_changed = QtCore.Signal()

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._setup_validators()
        self._connect_signals()

    def _setup_validators(self):
        self.ui.lineEdit_low_width.setValidator(QIntValidator(64, 4096, self))
        self.ui.lineEdit_low_height.setValidator(QIntValidator(64, 4096, self))
        self.ui.lineEdit_high_batch.setValidator(QIntValidator(1, 128, self))
        self.ui.lineEdit_low_batch.setValidator(QIntValidator(1, 128, self))

    def _connect_signals(self):
        pass

    def is_full_res_enabled(self):
        return self.ui.checkBox_enable_full_res.isChecked()

    def is_low_res_enabled(self):
        return self.ui.checkBox_enable_low_res.isChecked()

    def is_dual_output(self):
        return self.ui.checkBox_dual_output.isChecked()

    def is_strict_ffmpeg(self):
        return self.ui.checkBox_ffmpeg.isChecked()

    def get_full_batch_size(self):
        return int(self.ui.lineEdit_high_batch.text() or 10)

    def get_low_batch_size(self):
        return int(self.ui.lineEdit_low_batch.text() or 50)

    def get_low_width(self):
        return int(self.ui.lineEdit_low_width.text() or 1920)

    def get_low_height(self):
        return int(self.ui.lineEdit_low_height.text() or 1080)

    def set_full_res_enabled(self, enabled):
        self.ui.checkBox_enable_full_res.setChecked(enabled)

    def set_low_res_enabled(self, enabled):
        self.ui.checkBox_enable_low_res.setChecked(enabled)

    def set_dual_output(self, enabled):
        self.ui.checkBox_dual_output.setChecked(enabled)

    def set_strict_ffmpeg(self, enabled):
        self.ui.checkBox_ffmpeg.setChecked(enabled)

    def set_full_batch_size(self, value):
        self.ui.lineEdit_high_batch.setText(str(value))

    def set_low_batch_size(self, value):
        self.ui.lineEdit_low_batch.setText(str(value))

    def set_low_width(self, value):
        self.ui.lineEdit_low_width.setText(str(value))

    def set_low_height(self, value):
        self.ui.lineEdit_low_height.setText(str(value))


class ProcessingControlsWidget(QtWidgets.QWidget):
    start_processing = QtCore.Signal()
    stop_processing = QtCore.Signal()
    start_single = QtCore.Signal()
    update_sidecar = QtCore.Signal()

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._connect_signals()

    def _connect_signals(self):
        self.ui.pushButton_start.clicked.connect(self.start_processing.emit)
        self.ui.pushButton_stop.clicked.connect(self.stop_processing.emit)
        self.ui.pushButton_single.clicked.connect(self.start_single.emit)
        self.ui.pushButton_update_sidecar.clicked.connect(self.update_sidecar.emit)

    def set_status(self, text):
        self.ui.label_status.setText(text)

    def get_status(self):
        return self.ui.label_status.text()

    def set_progress(self, value):
        self.ui.progressBar.setValue(value)

    def get_progress(self):
        return self.ui.progressBar.value()

    def set_progress_max(self, max):
        self.ui.progressBar.setMaximum(max)

    def get_progress_max(self):
        return self.ui.progressBar.maximum()

    def set_enabled(self, enabled):
        self.ui.pushButton_start.setEnabled(enabled)
        self.ui.pushButton_single.setEnabled(enabled)
        self.ui.pushButton_stop.setEnabled(not enabled)

    def set_filename(self, filename):
        self.ui.label_info_filename_value.setText(filename)


class VideoListWidget(QtWidgets.QWidget):
    load_clicked = QtCore.Signal()

    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui
        self._connect_signals()

    def _connect_signals(self):
        self.ui.pushButton_load_refresh.clicked.connect(self._on_load)

    def _on_load(self):
        self.load_clicked.emit()

    def get_video_count(self):
        text = self.ui.label_video_info.text()
        if "/" in text:
            parts = text.split("/")
            if len(parts) >= 2:
                try:
                    return int(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 0

    def get_current_video_idx(self):
        try:
            return int(self.ui.label_video_info.text().split(":")[-1].split("/")[0].strip()) - 1
        except (ValueError, IndexError):
            return 0

    def set_video_info(self, current, total):
        self.ui.label_video_info.setText(f"Video: {current + 1} / {total}")

    def set_frame_info(self, current, total):
        self.ui.label_frame_info.setText(f"Frame: {current + 1} / {total}")

    def set_map_selector_items(self, items):
        self.ui.comboBox_map_select.clear()
        self.ui.comboBox_map_select.addItems(items)

    def get_selected_map(self):
        return self.ui.comboBox_map_select.currentText()


class DevToolsWidget(QtWidgets.QWidget):
    def __init__(self, ui, parent=None):
        super().__init__(parent)
        self.ui = ui

    def is_splat_test(self):
        return self.ui.checkBox_splat_test.isChecked()

    def is_map_test(self):
        return self.ui.checkBox_map_test.isChecked()

    def set_splat_test(self, enabled):
        self.ui.checkBox_splat_test.setChecked(enabled)

    def set_map_test(self, enabled):
        self.ui.checkBox_map_test.setChecked(enabled)
        text = self.ui.label_video_info.text()
        if "/" in text:
            parts = text.split("/")
            if len(parts) >= 2:
                try:
                    return int(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        return 0

    def get_current_video_idx(self):
        try:
            return int(self.ui.label_video_info.text().split(":")[-1].split("/")[0].strip()) - 1
        except (ValueError, IndexError):
            return 0

    def set_video_info(self, current, total):
        self.ui.label_video_info.setText(f"Video: {current + 1} / {total}")

    def set_frame_info(self, current, total):
        self.ui.label_frame_info.setText(f"Frame: {current + 1} / {total}")

    def set_map_selector_items(self, items):
        self.ui.comboBox_map_select.clear()
        self.ui.comboBox_map_select.addItems(items)

    def get_selected_map(self):
        return self.ui.comboBox_map_select.currentText()
