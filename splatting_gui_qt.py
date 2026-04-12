import sys
import logging
import json
import os
import numpy as np
import cv2
from PySide6 import QtWidgets, QtGui, QtCore
from core.ui.splatting_ui import Ui_MainWindow
from core.ui.splatting_widgets import (
    InputOutputWidget,
    PreviewControlsWidget,
    StereoProjectionWidget,
    DepthPreprocessingWidget,
    SplattingSettingsWidget,
    ProcessingResolutionWidget,
    ProcessingControlsWidget,
    VideoListWidget,
    DevToolsWidget,
)
from core.ui.preview_controller import PreviewController
from core.splatting.controller import SplattingController
from core.splatting.batch_processing import ProcessingSettings
from core.ui.workers.preview_render_worker import PreviewRenderWorker
from core.ui.workers.playback_worker import PlaybackWorker
from PIL.ImageQt import ImageQt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreviewWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Splatting Preview")
        self.layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("No Video Loaded")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout.addWidget(self.label)
        self.resize(800, 600)
        self._main_window = parent
        self._current_pixmap = None
        self._scale_factor = None

    def set_scale_factor(self, factor):
        self._scale_factor = factor
        if self._current_pixmap:
            self._update_display()

    def _get_scaled_size(self):
        if self._current_pixmap is None:
            return self.label.size()
        if self._scale_factor is None:
            return self.label.size()
        else:
            w = int(self._current_pixmap.width() * self._scale_factor)
            h = int(self._current_pixmap.height() * self._scale_factor)
            return QtCore.QSize(w, h)

    def _update_display(self):
        if self._current_pixmap:
            target_size = self._get_scaled_size()
            self.label.setPixmap(
                self._current_pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

    def closeEvent(self, event):
        if self._main_window and hasattr(self._main_window, "_wiggle_timer"):
            self._main_window._wiggle_timer.stop()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def set_image(self, pixmap):
        self._current_pixmap = pixmap
        self._update_display()


class SplattingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.controller = PreviewController()
        self.batch_controller = SplattingController()
        self.preview_window = PreviewWindow(self)

        self.current_video_idx = 0

        self._wiggle_state = True
        self._wiggle_timer = QtCore.QTimer(self)
        self._wiggle_timer.timeout.connect(self._wiggle_step)
        self._wiggle_left_img = None
        self._wiggle_right_img = None
        self._auto_save_sidecar = False
        self._encoding_config = {}

        self._init_widgets()
        self._setup_workers()
        self._setup_playback_timer()
        self._setup_processing()
        self._connect_signals()
        self.init_ui_defaults()
        self.load_config()

    def _init_widgets(self):
        self.io_widget = InputOutputWidget(self.ui, self)
        self.preview_widget = PreviewControlsWidget(self.ui, self)
        self.stereo_widget = StereoProjectionWidget(self.ui, self)
        self.depth_widget = DepthPreprocessingWidget(self.ui, self)
        self.splatting_widget = SplattingSettingsWidget(self.ui, self)
        self.resolution_widget = ProcessingResolutionWidget(self.ui, self)
        self.processing_widget = ProcessingControlsWidget(self.ui, self)
        self.video_list_widget = VideoListWidget(self.ui, self)
        self.dev_tools_widget = DevToolsWidget(self.ui, self)

    def _setup_workers(self):
        self.render_thread = QtCore.QThread(self)
        self.render_worker = PreviewRenderWorker(self.controller)
        self.render_worker.moveToThread(self.render_thread)
        self.render_worker.frame_ready.connect(self._on_frame_ready)
        self.render_worker.error.connect(self._on_render_error)
        self.render_thread.started.connect(lambda: logger.debug("Render thread started"))
        self.render_thread.start()

        self.playback_worker = PlaybackWorker()
        self.playback_worker.frame_advanced.connect(self._on_frame_advanced)
        self.playback_worker.playback_finished.connect(self._on_playback_finished)
        self.playback_worker.playback_started.connect(
            lambda: self.ui.pushButton_play.setIcon(QtGui.QIcon.fromTheme("media-playback-pause"))
        )
        self.playback_worker.playback_stopped.connect(
            lambda: self.ui.pushButton_play.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))
        )

    def _setup_playback_timer(self):
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.playback_worker.tick)
        self._is_fast_forward = False

    def _setup_processing(self):
        self._processing_timer = QtCore.QTimer(self)
        self._processing_timer.timeout.connect(self._check_batch_progress)
        self._processing_timer.start(100)

    def _check_batch_progress(self):
        progress = self.batch_controller.get_progress()
        if progress is None:
            return
        if progress == "finished":
            self._processing_timer.stop()
            self.processing_widget.set_status("Processing complete")
            self._enable_inputs(True)
            return
        if isinstance(progress, tuple):
            msg_type = progress[0]
            if msg_type == "status":
                self.processing_widget.set_status(progress[1])
            elif msg_type == "total":
                self.processing_widget.set_progress_max(progress[1])
                self.processing_widget.set_progress(0)
            elif msg_type == "processed":
                self.processing_widget.set_progress(progress[1])
            elif msg_type == "update_info":
                self.processing_widget.set_filename(progress[1].get("filename", ""))
            elif msg_type == "diagnostic_capture":
                self._handle_diagnostic_capture(progress[1])

    def closeEvent(self, event):
        self.save_config()
        self.play_timer.stop()
        self.batch_controller.stop()
        self.render_worker.stop()
        self.render_thread.quit()
        self.render_thread.wait(2000)
        self.controller.cleanup()
        self.preview_window.close()
        event.accept()

    def save_config(self):
        config = self.get_params()
        config["input_source"] = self.io_widget.get_source_path()
        config["input_depth"] = self.io_widget.get_depth_path()
        config["output_splatted"] = self.io_widget.get_output_path()
        config["sidecar_path"] = self.io_widget.get_sidecar_path()
        config["multi_map"] = self.io_widget.is_multi_map()
        config["low_width"] = str(self.resolution_widget.get_low_width())
        config["low_height"] = str(self.resolution_widget.get_low_height())
        config["enable_full_res"] = self.resolution_widget.is_full_res_enabled()
        config["enable_low_res"] = self.resolution_widget.is_low_res_enabled()
        config["dual_output"] = self.resolution_widget.is_dual_output()
        config["strict_ffmpeg"] = self.resolution_widget.is_strict_ffmpeg()
        config["high_batch"] = str(self.resolution_widget.get_full_batch_size())
        config["low_batch"] = str(self.resolution_widget.get_low_batch_size())
        config["process_length"] = str(self.splatting_widget.get_process_length())
        config["slider_disparity"] = self.stereo_widget.get_disparity()
        config["slider_convergence"] = self.stereo_widget.get_convergence()
        config["slider_gamma"] = self.stereo_widget.get_gamma()
        config["slider_border_width"] = self.stereo_widget.get_border_width()
        config["slider_bias"] = self.stereo_widget.get_border_bias()
        config["cross_view"] = self.stereo_widget.is_cross_view()
        config["resume"] = self.stereo_widget.is_resume_enabled()
        config["preview_scale"] = self.preview_widget.get_scale()
        config["debug_logging"] = self.ui.action_debug.isChecked()
        config["auto_update_sidecar"] = self.ui.action_auto_update_sidecar.isChecked()
        config["encoding"] = self._encoding_config

        try:
            with open("config_splat.splatcfg", "w") as f:
                json.dump(config, f, indent=4)
            logger.info("Configuration saved.")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def load_config(self):
        if not os.path.exists("config_splat.splatcfg"):
            return

        try:
            with open("config_splat.splatcfg", "r") as f:
                config = json.load(f)

            self.io_widget.set_source_path(config.get("input_source", "./workspace/clips"))
            self.io_widget.set_depth_path(config.get("input_depth", "./workspace/depth"))
            self.io_widget.set_output_path(config.get("output_splatted", "./workspace/splat"))
            self.io_widget.set_sidecar_path(str(config.get("sidecar_path", "./workspace/sidecar")))
            self.io_widget.set_multi_map(config.get("multi_map", False))
            self.resolution_widget.set_low_width(config.get("low_width", 640))
            self.resolution_widget.set_low_height(config.get("low_height", 320))
            self.resolution_widget.set_full_res_enabled(config.get("enable_full_res", True))
            self.resolution_widget.set_low_res_enabled(config.get("enable_low_res", True))
            self.resolution_widget.set_dual_output(config.get("dual_output", False))
            self.resolution_widget.set_strict_ffmpeg(config.get("strict_ffmpeg", False))
            self.resolution_widget.set_full_batch_size(config.get("high_batch", 10))
            self.resolution_widget.set_low_batch_size(config.get("low_batch", 50))
            self.splatting_widget.set_process_length(config.get("process_length", 0))
            self.splatting_widget.set_mesh_extrusion(config.get("mesh_extrusion", 0.5))
            self.splatting_widget.set_mesh_density(config.get("mesh_density", 0.5))
            self.splatting_widget.set_mesh_bias(config.get("mesh_bias", 0.5))
            self.splatting_widget.set_mesh_dolly(config.get("mesh_dolly", 0.0))

            self.ui.centralwidget.blockSignals(True)
            try:

                def val(key, default):
                    return int(float(config.get(key, default)))

                self.stereo_widget.set_disparity(val("max_disp", 35))
                self.stereo_widget.set_convergence(config.get("convergence_point", 1.0))
                self.stereo_widget.set_gamma(config.get("gamma", 1.0))
                self.stereo_widget.set_border_width(val("border_width", 0))
                self.stereo_widget.set_border_bias(config.get("border_bias", 0.5))
                self.stereo_widget.set_cross_view(config.get("cross_view", False))
                self.stereo_widget.set_resume_enabled(config.get("resume", False))
                self.depth_widget.set_dilate_x(val("dilate_x", 12))
                self.depth_widget.set_dilate_y(config.get("dilate_y", 3))
                self.depth_widget.set_blur_x(val("blur_x", 5))
                self.depth_widget.set_blur_y(val("blur_y", 5))

                preview_mode = config.get("preview_source", "Splat Result")
                self.preview_widget.set_preview_mode(preview_mode)

                preview_scale = config.get("preview_scale", "Auto")
                self.preview_widget.set_scale(preview_scale)

                self.ui.action_debug.setChecked(config.get("debug_logging", False))
                self.ui.action_auto_update_sidecar.setChecked(config.get("auto_update_sidecar", False))
                self._encoding_config = config.get("encoding", {})

            finally:
                self.ui.centralwidget.blockSignals(False)

            self.stereo_widget.update_labels()
            self.depth_widget.update_labels()
            logger.info("Configuration auto-loaded.")
        except Exception as e:
            self.ui.centralwidget.blockSignals(False)
            logger.error(f"Failed to load config: {e}")

    def init_ui_defaults(self):
        self.preview_widget.init_ui_defaults()
        self.stereo_widget.init_ui_defaults()
        self.ui.action_debug.setChecked(False)

    def _connect_signals(self):
        self.preview_widget.frame_changed.connect(self._on_frame_change)
        self.preview_widget.video_changed.connect(lambda delta: self.change_video(delta))
        self.preview_widget.playback_requested.connect(self.toggle_playback)
        self.preview_widget.scale_changed.connect(self._on_preview_scale_changed)

        self.processing_widget.start_processing.connect(self._on_start_processing)
        self.processing_widget.stop_processing.connect(self._on_stop_processing)
        self.processing_widget.start_single.connect(self._on_start_single_processing)
        self.processing_widget.update_sidecar.connect(self.save_sidecar)

        self.video_list_widget.load_clicked.connect(self.load_videos)

        # Connect value_changed signals to refresh preview and clear buffer
        self.stereo_widget.value_changed.connect(self._on_param_changed)
        self.depth_widget.value_changed.connect(self._on_param_changed)
        self.splatting_widget.value_changed.connect(self._on_param_changed)

        self.ui.comboBox_preview_source.currentTextChanged.connect(self._on_preview_source_changed)
        self.ui.spinBox_ff_speed.valueChanged.connect(self.update_playback_speed)
        self.ui.lineEdit_jump_to.editingFinished.connect(self.on_jump_to_clip)

        self.ui.action_debug.triggered.connect(self.on_debug_toggled)
        self.ui.action_calculator.triggered.connect(self.on_action_calculator)
        self.ui.action_load_settings.triggered.connect(self.on_load_settings)
        self.ui.action_save.triggered.connect(self.on_save_settings)
        self.ui.action_save_to_file.triggered.connect(self.on_save_settings_to_file)
        self.ui.action_fsexport_to_sidecar.triggered.connect(self.on_fsexport_to_sidecar)
        self.ui.action_update_from_sidecar.triggered.connect(self.on_update_from_sidecar)
        self.ui.action_auto_update_sidecar.triggered.connect(self.on_auto_update_sidecar)
        self.ui.action_encoder.triggered.connect(self.on_encoder_settings)
        self.ui.action_guide.triggered.connect(self.on_user_guide)
        self.ui.action_about.triggered.connect(self.on_about)
        self.ui.action_exit.triggered.connect(self.close)
        self.ui.action_restore_from_finished.triggered.connect(self.on_restore_finished)
        self.ui.action_load_fsexport.triggered.connect(self.on_load_fsexport)

    def _on_preview_source_changed(self, mode):
        self._request_render(self.preview_widget.get_frame())

    def _on_param_changed(self):
        """Handle parameter changes from stereo/depth widgets - refresh preview."""
        # Clear the buffer to force re-render with new params
        self.controller.buffer.clear()
        # Re-render current frame
        self._request_render(self.preview_widget.get_frame())

    def on_jump_to_clip(self):
        target = self.preview_widget.get_current_video_idx()
        if 0 <= target < len(self.controller.video_list):
            self.current_video_idx = target
            self.change_video(0)

    def toggle_playback(self, fast=False):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.playback_worker.stop()
        else:
            self._is_fast_forward = fast
            step = self.ui.spinBox_ff_speed.value() if fast else 1
            self.playback_worker.set_step(step)
            self.playback_worker.start()
            interval = 41 if not fast else 30
            self.play_timer.start(interval)

    def update_playback_speed(self, is_fast=None):
        if not self.play_timer.isActive():
            return

        interval = 80
        if is_fast or (is_fast is None and self.ui.pushButton_play.text() == "⏸"):
            speed = max(1, self.ui.spinBox_ff_speed.value())
            interval = max(30, 1000 // (speed + 10))

        self.play_timer.setInterval(interval)
        self.playback_worker.set_step(self.ui.spinBox_ff_speed.value())

    def _on_frame_advanced(self, frame_idx):
        self.ui.horizontalSlider.blockSignals(True)
        self.ui.horizontalSlider.setValue(frame_idx)
        self.ui.horizontalSlider.blockSignals(False)
        self._request_render(frame_idx)

    def _on_playback_finished(self):
        self.play_timer.stop()
        self.ui.pushButton_play.setIcon(QtGui.QIcon.fromTheme("media-playback-start"))

    def _on_frame_ready(self, pil_img):
        if isinstance(pil_img, tuple):
            self._wiggle_left_img, self._wiggle_right_img = pil_img
            if not self._wiggle_timer.isActive() and self.preview_window.isVisible():
                self._wiggle_timer.start(60)
            self._show_wiggle_frame(self._wiggle_state)
            return

        self._wiggle_timer.stop()
        self._wiggle_left_img = None
        self._wiggle_right_img = None

        if pil_img and self.preview_window.isVisible():
            pixmap = QtGui.QPixmap.fromImage(ImageQt(pil_img))
            self.preview_window.set_image(pixmap)
            self._save_sidecar_async()

    def _wiggle_step(self):
        self._wiggle_state = not self._wiggle_state
        self._show_wiggle_frame(self._wiggle_state)

    def _show_wiggle_frame(self, show_left: bool):
        if show_left and self._wiggle_left_img:
            pixmap = QtGui.QPixmap.fromImage(ImageQt(self._wiggle_left_img))
            self.preview_window.set_image(pixmap)
        elif not show_left and self._wiggle_right_img:
            pixmap = QtGui.QPixmap.fromImage(ImageQt(self._wiggle_right_img))
            self.preview_window.set_image(pixmap)

    def _on_render_error(self, error_msg):
        logger.error(f"Render error: {error_msg}")

    def _request_render(self, frame_idx=None):
        if not self.controller.video_list:
            return
        if frame_idx is None:
            frame_idx = self.ui.horizontalSlider.value()

        try:
            frame_idx = int(float(frame_idx))
        except (ValueError, TypeError):
            frame_idx = self.ui.horizontalSlider.value()

        if frame_idx < 0:
            return
        params = self._get_render_params(frame_idx)
        if params:
            self.render_worker.render_frame(frame_idx, params)

    def _on_frame_change(self, frame):
        total = self.controller.get_total_frames()
        self.video_list_widget.set_frame_info(frame, total)
        self._request_render(frame)

    def _on_preview_scale_changed(self, value: str):
        if not value or value == "Auto":
            self.preview_window.set_scale_factor(None)
        elif value.endswith("%"):
            scale_percent = int(value.rstrip("%"))
            self.preview_window.set_scale_factor(scale_percent / 100.0)

    def on_debug_toggled(self, checked: bool):
        level = logging.DEBUG if checked else logging.INFO
        logging.getLogger().setLevel(level)
        logger.debug(f"Debug logging {'enabled' if checked else 'disabled'}")

    def _get_render_params(self, frame_idx=None) -> dict:
        params = self.get_params()
        if params:
            idx = frame_idx if frame_idx is not None else self.ui.horizontalSlider.value()
            params["wiggle_toggle"] = idx % 2 == 0
            params["mode"] = params.get("preview_source", "Splat Result")
        return params

    def get_sidecar_path(self):
        return self.controller.get_sidecar_path()

    def _save_sidecar_async(self):
        if self._auto_save_sidecar:
            params = self.get_params()
            self.controller.save_sidecar(params)

    def save_sidecar(self):
        params = self.get_params()
        self.controller.save_sidecar(params)

    def _load_sidecar(self):
        if not self._auto_save_sidecar:
            return

        sidecar_data = self.controller.load_sidecar()
        if not sidecar_data:
            return

        current_mode = self.preview_widget.get_preview_mode()

        self.ui.centralwidget.blockSignals(True)
        try:
            params = self.controller._map_sidecar_to_params(sidecar_data)

            def val(key, default):
                v = params.get(key, default)
                return int(float(v)) if v is not None else int(default)

            self.ui.horizontalSlider_disparity.blockSignals(True)
            self.ui.horizontalSlider_convergence.blockSignals(True)
            self.ui.horizontalSlider_gamma.blockSignals(True)
            self.ui.horizontalSlider_border_bias.blockSignals(True)
            self.ui.horizontalSlider_dilate_x.blockSignals(True)
            self.ui.horizontalSlider_dilate_y.blockSignals(True)
            self.ui.horizontalSlider_blur_x.blockSignals(True)
            self.ui.horizontalSlider_blur_y.blockSignals(True)

            sidecar_mode = params.get("preview_source", current_mode)
            self.preview_widget.set_preview_mode(sidecar_mode)

            self.depth_widget.set_blur_bias(sidecar_data.get("blur_bias", 0.5))
            self.splatting_widget.set_mesh_extrusion(sidecar_data.get("mesh_extrusion", 0.5))
            self.splatting_widget.set_mesh_density(sidecar_data.get("mesh_density", 0.5))
            self.splatting_widget.set_mesh_dolly(sidecar_data.get("mesh_dolly", 0.0))

            steering = float(params.get("view_bias", 0.0))
            self.splatting_widget.set_mesh_bias(steering)

            border_bias = float(params.get("border_bias", 0.0))
            self.stereo_widget.set_border_bias(border_bias)
            self.stereo_widget.update_labels()

            self.stereo_widget.set_disparity(val("max_disp", 35))
            self.stereo_widget.set_convergence(params.get("convergence_point", 0.5))
            self.stereo_widget.set_gamma(params.get("gamma", 1.0))
            self.depth_widget.set_dilate_x(val("dilate_x", 0))
            self.depth_widget.set_dilate_y(val("dilate_y", 0))
            self.depth_widget.set_blur_x(val("blur_x", 5))
            self.depth_widget.set_blur_y(val("blur_y", 5))

            if "cross_view" in params:
                self.stereo_widget.set_cross_view(bool(params["cross_view"]))

        finally:
            self.ui.horizontalSlider_disparity.blockSignals(False)
            self.ui.horizontalSlider_convergence.blockSignals(False)
            self.ui.horizontalSlider_gamma.blockSignals(False)
            self.ui.horizontalSlider_border_bias.blockSignals(False)
            self.ui.horizontalSlider_dilate_x.blockSignals(False)
            self.ui.horizontalSlider_dilate_y.blockSignals(False)
            self.ui.horizontalSlider_blur_x.blockSignals(False)
            self.ui.horizontalSlider_blur_y.blockSignals(False)
            self.ui.lineEdit_mesh_bias.blockSignals(False)
            self.ui.lineEdit_blur_bias.blockSignals(False)
            self.ui.centralwidget.blockSignals(False)

        self.stereo_widget.update_labels()
        self.depth_widget.update_labels()
        logger.info(f"Sidecar loaded for video {self.current_video_idx + 1}")

    def load_sidecar(self):
        self._load_sidecar()

    def get_params(self) -> dict:
        try:
            preview_mode = self.preview_widget.get_preview_mode()
            modes_without_cross_view = ["Original (Left Eye)", "Splat Result", "Splat Result(Low)"]

            logger.debug(f"get_params called, preview_mode: {preview_mode}")

            params = {
                "preview_source": preview_mode,
                "strict_ffmpeg_decode": self.resolution_widget.is_strict_ffmpeg(),
                "max_disp": self.stereo_widget.get_disparity(),
                "convergence_point": self.stereo_widget.get_convergence(),
                "gamma": self.stereo_widget.get_gamma(),
                "border_mode": self.stereo_widget.get_border_mode(),
                "border_width": self.stereo_widget.get_border_width(),
                "view_bias": self.splatting_widget.get_mesh_bias(),
                "border_bias": self.stereo_widget.get_border_bias(),
                "cross_view": self.stereo_widget.is_cross_view()
                if preview_mode not in modes_without_cross_view
                else False,
                "dilate_x": self.depth_widget.get_dilate_x(),
                "dilate_y": self.depth_widget.get_dilate_y(),
                "blur_x": self.depth_widget.get_blur_x(),
                "blur_y": self.depth_widget.get_blur_y(),
                "blur_bias": self.depth_widget.get_blur_bias(),
                "mesh_extrusion": self.splatting_widget.get_mesh_extrusion(),
                "mesh_density": self.splatting_widget.get_mesh_density(),
                "mesh_dolly": self.splatting_widget.get_mesh_dolly(),
                "mask_type": self.splatting_widget.get_mask_type(),
                "slider_disparity": self.stereo_widget.get_disparity(),
                "slider_convergence": self.stereo_widget.get_convergence(),
                "slider_bias": self.stereo_widget.get_border_bias(),
            }
            logger.debug(f"get_params returning: {params}")
            return params
        except Exception as e:
            logger.error(f"get_params exception: {e}")
            return {}

    def on_action_calculator(self):
        import subprocess

        base_dir = os.path.dirname(os.path.abspath(__file__))
        calculator_path = os.path.join(base_dir, "..", "scripts", "ResCalc_v3.html")
        calculator_path = os.path.normpath(calculator_path)
        if os.path.exists(calculator_path):
            subprocess.Popen(["start", "", calculator_path], shell=True)
        else:
            logger.error(f"Calculator not found: {calculator_path}")

    def _get_processing_settings(self) -> ProcessingSettings:
        settings = ProcessingSettings(
            input_source_clips=self.io_widget.get_source_path(),
            input_depth_maps=self.io_widget.get_depth_path(),
            output_splatted=self.io_widget.get_output_path(),
            max_disp=self.stereo_widget.get_disparity(),
            zero_disparity_anchor=self.stereo_widget.get_convergence(),
            depth_gamma=self.stereo_widget.get_gamma(),
            process_length=self.splatting_widget.get_process_length(),
            enable_full_resolution=self.resolution_widget.is_full_res_enabled(),
            full_res_batch_size=self.resolution_widget.get_full_batch_size(),
            enable_low_resolution=self.resolution_widget.is_low_res_enabled(),
            low_res_width=self.resolution_widget.get_low_width(),
            low_res_height=self.resolution_widget.get_low_height(),
            low_res_batch_size=self.resolution_widget.get_low_batch_size(),
            dual_output=self.resolution_widget.is_dual_output(),
            strict_ffmpeg_decode=self.resolution_widget.is_strict_ffmpeg(),
            move_to_finished=self.stereo_widget.is_resume_enabled(),
            sidecar_folder=self.io_widget.get_sidecar_path(),
            multi_map=self.io_widget.is_multi_map(),
            depth_dilate_size_x=self.depth_widget.get_dilate_x(),
            depth_dilate_size_y=self.depth_widget.get_dilate_y(),
            depth_blur_size_x=self.depth_widget.get_blur_x(),
            depth_blur_size_y=self.depth_widget.get_blur_y(),
            mask_mode=self.splatting_widget.get_mask_type(),
            is_test_mode=self.dev_tools_widget.is_splat_test() or self.dev_tools_widget.is_map_test(),
            test_target_frame_idx=self.preview_widget.get_frame()
            if self.dev_tools_widget.is_splat_test() or self.dev_tools_widget.is_map_test()
            else None,
            test_type="map" if self.dev_tools_widget.is_map_test() else "splat",
        )
        return settings

    def _enable_inputs(self, enabled: bool):
        self.processing_widget.set_enabled(enabled)

    def _on_start_processing(self):
        self._enable_inputs(False)
        self.batch_controller.stop_event.clear()
        settings = self._get_processing_settings()
        logger.info(
            f"[DEBUG] Starting batch with test_target_frame_idx={settings.test_target_frame_idx}, is_test_mode={settings.is_test_mode}"
        )
        logger.info(
            f"[DEBUG] full_res_batch_size={settings.full_res_batch_size}, low_res_batch_size={settings.low_res_batch_size}"
        )
        self.batch_controller.start_batch(settings)
        self._processing_timer.start(100)

    def _on_stop_processing(self):
        self.batch_controller.stop()
        self.processing_widget.set_status("Stopping...")

    def _on_start_single_processing(self):
        self._enable_inputs(False)
        self.batch_controller.stop_event.clear()
        settings = self._get_processing_settings()
        self.batch_controller.start_batch(
            settings, from_index=self.current_video_idx, to_index=self.current_video_idx + 1
        )
        self._processing_timer.start(100)

    def on_load_settings(self):
        from core.splatting.config_manager import load_settings_from_file

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Settings from File", "", "Splat Config (*.splatcfg);;JSON files (*.json)"
        )
        if not filename:
            return
        try:
            config = load_settings_from_file(filename)
            self._apply_config_to_ui(config)
            self.processing_widget.set_status("Settings loaded.")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def on_save_settings(self):
        # config = self.get_params()
        self.save_config()
        self.processing_widget.set_status("Settings saved.")

    def on_save_settings_to_file(self):
        from core.splatting.config_manager import save_settings_to_file

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Settings to File", "", "Splat Config (*.splatcfg);;JSON files (*.json)"
        )
        if not filename:
            return
        try:
            config = self.get_params()
            save_settings_to_file(config, filename)
            self.processing_widget.set_status(f"Saved settings to {os.path.basename(filename)}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def on_fsexport_to_sidecar(self):
        from core.splatting.fusion_export import FusionSidecarGenerator

        sidecar_path = self.controller.get_sidecar_path()
        if not sidecar_path:
            return
        generator = FusionSidecarGenerator()
        generator.generate_sidecar(sidecar_path, self.get_params())
        self.processing_widget.set_status("FSExport sidecar generated.")

    def on_update_from_sidecar(self):
        self._load_sidecar()
        self.processing_widget.set_status("Loaded sidecar data.")

    def on_auto_update_sidecar(self, checked):
        self._auto_save_sidecar = checked
        self.ui.action_auto_update_sidecar.setChecked(checked)

    def on_encoder_settings(self):
        from core.ui.qt_encoding_settings import QtEncodingSettingsDialog

        dialog = QtEncodingSettingsDialog(self, app_config={"encoding": self._encoding_config})
        if dialog.exec():
            self._encoding_config = dialog.get_config()
            self.save_config()
            self.processing_widget.set_status("Encoding settings saved")

    def on_user_guide(self):
        import webbrowser

        webbrowser.open("https://github.com/Teng0/StereoCrafter")

    def on_about(self):
        QtWidgets.QMessageBox.about(
            self,
            "About StereoCrafter",
            "StereoCrafter Splatting (Qt)\n\n"
            "A tool for generating right-eye stereo views from source video and depth maps.\n"
            "Based on PyTorch, Diffusers, and Qt.\n\n"
            "(C) 2024 Some Rights Reserved",
        )

    def on_restore_finished(self):
        from core.common.file_organizer import move_files_to_finished

        reply = QtWidgets.QMessageBox.question(
            self,
            "Restore Finished Files",
            "Are you sure you want to move all files from 'finished' folders back to their input directories?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.No:
            return

        source_clip_dir = self.io_widget.get_source_path()
        depth_map_dir = self.io_widget.get_depth_path()

        if not (os.path.isdir(source_clip_dir) and os.path.isdir(depth_map_dir)):
            QtWidgets.QMessageBox.warning(
                self,
                "Restore Error",
                "Restore 'finished' operation is only applicable when Input Source Clips and Input Depth Maps are set to directories (batch mode).",
            )
            return

        files_to_move = []
        finished_source_folder = os.path.join(source_clip_dir, "finished")
        finished_depth_folder = os.path.join(depth_map_dir, "finished")

        if os.path.isdir(finished_source_folder):
            for filename in os.listdir(finished_source_folder):
                src_path = os.path.join(finished_source_folder, filename)
                if os.path.isfile(src_path):
                    files_to_move.append((src_path, source_clip_dir))

        if os.path.isdir(finished_depth_folder):
            for filename in os.listdir(finished_depth_folder):
                src_path = os.path.join(finished_depth_folder, filename)
                if os.path.isfile(src_path):
                    files_to_move.append((src_path, depth_map_dir))

        if not files_to_move:
            QtWidgets.QMessageBox.information(
                self, "Restore Complete", "No files found in 'finished' folders to restore."
            )
            return

        moved, failed, _ = move_files_to_finished(files_to_move, logger)

        QtWidgets.QMessageBox.information(
            self,
            "Restore Complete",
            f"Finished files restoration attempted.\n{moved} files moved.\n{failed} errors occurred.",
        )

    def on_load_fsexport(self):
        from core.splatting.fusion_export import FusionSidecarGenerator

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Fusion Export", "", "Fusion Export (*.fsexport);;All files (*)"
        )
        if not filename:
            return
        try:
            generator = FusionSidecarGenerator()
            data = generator.load_fsexport(filename)
            if data:
                self._apply_config_to_ui(data)
                self.processing_widget.set_status(f"Loaded Fusion export: {filename}")
            else:
                QtWidgets.QMessageBox.warning(self, "Load Failed", "Could not load Fusion export file.")
        except Exception as e:
            logger.error(f"Failed to load fsexport: {e}")
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load: {e}")

    def _apply_config_to_ui(self, config: dict):
        if "max_disp" in config:
            self.stereo_widget.set_disparity(int(config["max_disp"]))
        if "convergence_point" in config:
            self.stereo_widget.set_convergence(float(config["convergence_point"]))
        if "gamma" in config:
            self.stereo_widget.set_gamma(float(config["gamma"]))
        if "slider_border_width" in config:
            self.stereo_widget.set_border_width(config["slider_border_width"])
        if "slider_bias" in config:
            self.stereo_widget.set_border_bias(config["slider_bias"])
        if "input_source" in config:
            self.io_widget.set_source_path(config["input_source"])
        if "input_depth" in config:
            self.io_widget.set_depth_path(config["input_depth"])
        if "output_splatted" in config:
            self.io_widget.set_output_path(config["output_splatted"])
        if "sidecar_path" in config:
            self.io_widget.set_sidecar_path(config["sidecar_path"])
        if "preview_source" in config:
            self.preview_widget.set_preview_mode(config["preview_source"])
        self.stereo_widget.update_labels()
        self.depth_widget.update_labels()

    def load_videos(self):
        src = self.io_widget.get_source_path()
        depth = self.io_widget.get_depth_path()
        sidecar = self.io_widget.get_sidecar_path()
        multi = self.io_widget.is_multi_map()

        self.controller.set_sidecar_folder(sidecar)
        self.batch_controller.sidecar_folder = sidecar
        self.batch_controller.input_source_clips = src
        self.batch_controller.input_depth_maps = depth
        self.batch_controller.output_splatted = self.io_widget.get_output_path()
        self.batch_controller.video_list = self.controller.video_list

        old_path = None
        if self.controller.video_list:
            old_path = self.controller.video_list[self.current_video_idx]["source_video"]

        video_list = self.controller.load_video_list(src, depth, multi)
        if video_list:
            new_idx = 0
            if old_path:
                for idx, entry in enumerate(video_list):
                    if entry["source_video"] == old_path:
                        new_idx = idx
                        break

            self.current_video_idx = new_idx
            self.change_video(0)
            self.processing_widget.set_status(f"Loaded {len(video_list)} videos.")

            if not self.preview_window.isVisible():
                self.preview_window.show()
                self._request_render(0)

    def change_video(self, delta):
        new_idx = self.current_video_idx + delta
        params = self.get_params()
        logger.debug(f"change_video called with delta={delta}, params={params}")
        if not params:
            params = {"strict_ffmpeg_decode": False}
        if self.controller.set_current_video(new_idx, params):
            self.current_video_idx = new_idx

            self._load_sidecar()

            entry = self.controller.get_current_video_entry()
            if entry:
                logger.info(f"Loading: {entry['source_video']}")

            total_frames = self.controller.get_total_frames()
            self.playback_worker.set_total_frames(total_frames)

            self.preview_widget.set_total_frames(total_frames)
            self.preview_widget.set_frame(0)

            if self.io_widget.is_multi_map() and entry:
                map_folder = os.path.basename(os.path.dirname(entry["depth_map"]))
                self.video_list_widget.set_map_selector_items([map_folder])
            else:
                self.video_list_widget.set_map_selector_items(["Default"])

            self.video_list_widget.set_video_info(new_idx, len(self.controller.video_list))
            self.video_list_widget.set_frame_info(0, total_frames)
            self._request_render(0)
        else:
            logger.error(f"Could not load video index {new_idx}")

    def _handle_diagnostic_capture(self, payload: dict):
        """Handle diagnostic capture - save preview vs render comparison."""
        try:
            grid = payload.get("grid")
            task_name = payload.get("task_name", "render")
            test_type = payload.get("test_type", "splat")
            # task_suffix = payload.get("task_suffix", "")

            if grid is None:
                return

            # Determine output directory
            output_dir = self.io_widget.get_output_path()
            if not output_dir:
                output_dir = os.getcwd()
            os.makedirs(output_dir, exist_ok=True)

            # Get frame index
            frame_idx = self.preview_widget.get_frame()

            # Get video base name
            video_entry = self.controller.get_current_video_entry() if self.controller.video_list else {}
            video_name = os.path.splitext(os.path.basename(video_entry.get("source_video", "capture")))[0]

            # Get task suffix
            task_suffix_str = ""
            if task_name.lower() in ("hires", "full", "fullres"):
                task_suffix_str = "_hi"
            elif task_name.lower() in ("lowres", "low"):
                task_suffix_str = "_low"

            # Test type
            kind = "maptest" if test_type.startswith("map") else "splattest"

            # Create filenames
            stem = f"{video_name}_frame_{frame_idx:05d}_{kind}"

            # Save render grid as PNG
            grid_uint8 = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
            grid_bgr = cv2.cvtColor(grid_uint8, cv2.COLOR_RGB2BGR)
            render_path = os.path.join(output_dir, f"{stem}_render{task_suffix_str}.png")
            cv2.imwrite(render_path, grid_bgr)
            logger.info(f"Diagnostic render saved: {render_path}")

            # Capture preview from preview window if available
            if self.preview_window.isVisible() and self.preview_window.label.pixmap():
                preview_pixmap = self.preview_window.label.pixmap()
                preview_img = preview_pixmap.toImage()

                # Convert QImage to numpy array
                width = preview_img.width()
                height = preview_img.height()

                # Get the raw image data - Qt stores as BGRA
                # Convert to bytes and reshape
                # img_size = preview_img.sizeInBytes()
                ptr = preview_img.bits()

                # Create numpy array from the image data
                arr = np.array(ptr).reshape((height, width, 4))

                # Convert BGRA to BGR (Qt uses BGRA)
                preview_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

                preview_path = os.path.join(output_dir, f"{stem}_preview{task_suffix_str}.png")
                cv2.imwrite(preview_path, preview_bgr)
                logger.info(f"Diagnostic preview saved: {preview_path}")
            else:
                logger.warning("Preview window not available for capture")

            self.processing_widget.set_status(f"Test outputs saved to: {output_dir}")
        except Exception as e:
            logger.error(f"Diagnostic capture save failed: {e}", exc_info=True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SplattingApp()
    window.show()
    sys.exit(app.exec())
