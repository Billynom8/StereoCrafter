import sys
import logging
import json
import os
import shutil
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QDoubleValidator, QIntValidator
from core.ui.splatting_ui import Ui_MainWindow
from core.ui.preview_controller import PreviewController
from core.splatting.controller import SplattingController
from core.splatting.batch_processing import ProcessingSettings
from core.ui.workers.preview_render_worker import PreviewRenderWorker
from core.ui.workers.playback_worker import PlaybackWorker
from PIL.ImageQt import ImageQt

logging.basicConfig(level=logging.DEBUG)
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
        self._scale_factor = None  # None = auto (fit to window), float = fixed scale

    def set_scale_factor(self, factor):
        """Set a fixed scale factor (e.g., 1.0 = 100%, 0.5 = 50%). None for auto-fit."""
        self._scale_factor = factor
        if self._current_pixmap:
            self._update_display()

    def _get_scaled_size(self):
        """Calculate target size based on scale factor or auto-fit to window."""
        if self._current_pixmap is None:
            return self.label.size()

        if self._scale_factor is None:
            return self.label.size()
        else:
            w = int(self._current_pixmap.width() * self._scale_factor)
            h = int(self._current_pixmap.height() * self._scale_factor)
            return QtCore.QSize(w, h)

    def _update_display(self):
        """Update label with current pixmap scaled appropriately."""
        if self._current_pixmap:
            target_size = self._get_scaled_size()
            self.label.setPixmap(
                self._current_pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

    def closeEvent(self, event):
        """Stop wigglegram timer when preview window is closed."""
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

        # Wigglegram animation state
        self._wiggle_state = True  # True = show left, False = show right
        self._wiggle_timer = QtCore.QTimer(self)
        self._wiggle_timer.timeout.connect(self._wiggle_step)
        self._wiggle_left_img = None
        self._wiggle_right_img = None
        self._auto_save_sidecar = False
        self._encoding_config = {}

        self._setup_workers()
        self._setup_playback_timer()
        self._setup_processing()
        self.connect_signals()
        self.init_ui_defaults()
        self.load_config()

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
            self.ui.label_status.setText("Processing complete")
            self._enable_inputs(True)
            return
        if isinstance(progress, tuple):
            msg_type = progress[0]
            if msg_type == "status":
                self.ui.label_status.setText(progress[1])
            elif msg_type == "total":
                self.ui.progressBar.setMaximum(progress[1])
                self.ui.progressBar.setValue(0)
            elif msg_type == "processed":
                self.ui.progressBar.setValue(progress[1])
            elif msg_type == "update_info":
                self.ui.label_info_filename_value.setText(progress[1].get("filename", ""))

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
        config["input_source"] = self.ui.lineEdit_input_source.text()
        config["input_depth"] = self.ui.lineEdit_input_depth.text()
        config["output_splatted"] = self.ui.lineEdit_output_splatted.text()
        config["sidecar_path"] = self.ui.lineEdit_inout_sidecar.text()
        config["multi_map"] = self.ui.checkBox_multi_map.isChecked()
        config["low_width"] = str(self.ui.lineEdit_low_width.text())
        config["low_height"] = str(self.ui.lineEdit_low_height.text())
        config["process_length"] = str(self.ui.lineEdit_process_length.text())
        config["slider_disparity"] = self.ui.horizontalSlider_disparity.value()
        config["slider_convergence"] = self.ui.horizontalSlider_convergence.value()
        config["slider_gamma"] = self.ui.horizontalSlider_gamma.value()
        config["slider_border_width"] = self.ui.horizontalSlider_border_width.value()
        config["slider_bias"] = self.ui.horizontalSlider_border_bias.value()
        config["preview_scale"] = self.ui.comboBox_preview_scale.currentText()
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

            self.ui.lineEdit_input_source.setText(config.get("input_source", "./workspace/clips"))
            self.ui.lineEdit_input_depth.setText(config.get("input_depth", "./workspace/depth"))
            self.ui.lineEdit_output_splatted.setText(config.get("output_splatted", "./workspace/splat"))
            self.ui.lineEdit_inout_sidecar.setText(str(config.get("sidecar_path", "./workspace/sidecar")))
            self.ui.checkBox_multi_map.setChecked(config.get("multi_map", False))
            self.ui.lineEdit_low_width.setText(str(config.get("low_width", "640")))
            self.ui.lineEdit_low_height.setText(str(config.get("low_height", "320")))
            self.ui.lineEdit_process_length.setText(str(config.get("process_length", "0")))
            self.ui.lineEdit_mesh_extrusion.setText(str(config.get("mesh_extrusion", "0.5")))
            self.ui.lineEdit_mesh_density.setText(str(config.get("mesh_density", "0.5")))
            self.ui.lineEdit_mesh_bias.setText(str(config.get("mesh_bias", "0.5")))
            self.ui.lineEdit_mesh_dolly.setText(str(config.get("mesh_dolly", "0.0")))

            self.ui.centralwidget.blockSignals(True)
            try:
                self.ui.horizontalSlider_disparity.setValue(config.get("slider_disparity", 35))
                self.ui.horizontalSlider_convergence.setValue(config.get("slider_convergence", 100))
                self.ui.horizontalSlider_gamma.setValue(config.get("slider_gamma", 99))
                self.ui.horizontalSlider_border_width.setValue(config.get("slider_border_width", 0))
                self.ui.horizontalSlider_border_bias.setValue(config.get("slider_bias", 50))
                self.ui.checkBox_cross_view.setChecked(config.get("cross_view", False))

                def val(key, default):
                    return int(float(config.get(key, default)))

                self.ui.horizontalSlider_disparity.setValue(val("max_disp", 35))
                self.ui.horizontalSlider_convergence.setValue(int(float(config.get("convergence_point", 1.0)) * 100))
                self.ui.horizontalSlider_gamma.setValue(int(float(config.get("gamma", 1.0)) * 100))
                self.ui.horizontalSlider_border_width.setValue(val("border_width", 0))
                self.ui.horizontalSlider_border_bias.setValue(int(float(config.get("border_bias", 0.5)) * 100))
                self.ui.horizontalSlider_dilate_x.setValue(val("dilate_x", 12))
                self.ui.horizontalSlider_dilate_y.setValue(int(float(config.get("dilate_y", 3)) * 2.0))
                self.ui.horizontalSlider_blur_x.setValue(val("blur_x", 5))
                self.ui.horizontalSlider_blur_y.setValue(val("blur_y", 5))

                idx = self.ui.comboBox_preview_source.findText(config.get("preview_source", "Splat Result"))
                if idx >= 0:
                    self.ui.comboBox_preview_source.setCurrentIndex(idx)

                preview_scale = config.get("preview_scale", "Auto")
                if preview_scale:
                    idx = self.ui.comboBox_preview_scale.findText(preview_scale)
                    if idx >= 0:
                        self.ui.comboBox_preview_scale.setCurrentIndex(idx)
                        self.on_preview_scale_changed(preview_scale)

                self.ui.action_debug.setChecked(config.get("debug_logging", False))
                self.ui.action_auto_update_sidecar.setChecked(config.get("auto_update_sidecar", False))
                self._encoding_config = config.get("encoding", {})

            finally:
                self.ui.centralwidget.blockSignals(False)

            self.update_all_labels()
            logger.info("Configuration auto-loaded.")
        except Exception as e:
            self.ui.centralwidget.blockSignals(False)
            logger.error(f"Failed to load config: {e}")

    def init_ui_defaults(self):
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

        self.ui.lineEdit_mesh_extrusion.setValidator(QDoubleValidator(-1.0, 5.0, 3, self))
        self.ui.lineEdit_mesh_density.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        self.ui.lineEdit_mesh_bias.setValidator(QDoubleValidator(-1.0, 1.0, 3, self))
        self.ui.lineEdit_mesh_dolly.setValidator(QDoubleValidator(0.0, 100.0, 3, self))

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

        self.ui.horizontalSlider_border_bias.setMinimum(0)
        self.ui.horizontalSlider_border_bias.setMaximum(100)

        self.ui.lineEdit_low_width.setValidator(QIntValidator(64, 4096, self))
        self.ui.lineEdit_low_height.setValidator(QIntValidator(64, 4096, self))
        self.ui.lineEdit_high_batch.setValidator(QIntValidator(1, 128, self))
        self.ui.lineEdit_low_batch.setValidator(QIntValidator(1, 128, self))
        self.ui.lineEdit_process_length.setValidator(QIntValidator(0, 999999, self))
        self.ui.lineEdit_blur_bias.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        self.ui.lineEdit_jump_to.setValidator(QIntValidator(1, 999999, self))

        self.update_all_labels()

        self.ui.action_debug.setChecked(False)

    def setup_slider(self, slider, label, divisor=1.0):
        def on_change(value):
            if divisor == 0:
                label.setText(str(value))
            else:
                label.setText(f"{value / divisor:.2f}")
            self._request_render()

        slider.valueChanged.connect(on_change)

    def update_all_labels(self):
        self.ui.label_disparity_value.setText(str(self.ui.horizontalSlider_disparity.value()))
        self.ui.label_convergence_value.setText(f"{self.ui.horizontalSlider_convergence.value() / 100.0:.2f}")
        self.ui.label_gamma_value.setText(f"{self.ui.horizontalSlider_gamma.value() / 100.0:.2f}")
        self.ui.label_border_width_value.setText(str(self.ui.horizontalSlider_border_width.value()))
        bias_val = (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0
        self.ui.label_bias_value.setText(f"{bias_val:.2f}")

    def connect_signals(self):
        self.ui.pushButton_browse_source.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_input_source))
        self.ui.pushButton_browse_depth.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_input_depth))
        self.ui.pushButton_browse_output.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_output_splatted))
        self.ui.pushButton_browse_sidecar.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_inout_sidecar))
        self.ui.pushButton_load_refresh.clicked.connect(self.load_videos)

        self.ui.horizontalSlider.valueChanged.connect(self.on_frame_change)
        self.ui.pushButton_next.clicked.connect(lambda: self.change_video(1))
        self.ui.pushButton_prev.clicked.connect(lambda: self.change_video(-1))

        self.ui.pushButton_play.clicked.connect(self.toggle_playback)
        self.ui.pushButton_fast_forward.clicked.connect(lambda: self.toggle_playback(fast=True))
        self.ui.spinBox_ff_speed.valueChanged.connect(self.update_playback_speed)

        self.ui.comboBox_preview_scale.currentTextChanged.connect(self.on_preview_scale_changed)
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

        # Unconnected menu actions - add handlers
        self.ui.action_restore_from_finished.triggered.connect(self.on_restore_finished)
        self.ui.action_load_fsexport.triggered.connect(self.on_load_fsexport)

        self.ui.pushButton_start.clicked.connect(self.on_start_processing)
        self.ui.pushButton_stop.clicked.connect(self.on_stop_processing)
        self.ui.pushButton_single.clicked.connect(self.on_start_single_processing)
        self.ui.pushButton_update_sidecar.clicked.connect(self.save_sidecar)

    def on_bias_change(self, value):
        bias_val = (value - 50) / 50.0
        self.ui.label_bias_value.setText(f"{bias_val:.2f}")
        self._request_render()

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

    def on_loop_toggled(self, checked):
        self.playback_worker.set_loop_enabled(checked)

    def _on_frame_ready(self, pil_img):
        # Handle wigglegram tuple (left_img, right_img)
        if isinstance(pil_img, tuple):
            self._wiggle_left_img, self._wiggle_right_img = pil_img
            # Start wiggle timer if not already running
            if not self._wiggle_timer.isActive() and self.preview_window.isVisible():
                self._wiggle_timer.start(60)  # 60ms interval
            # Show first frame
            self._show_wiggle_frame(self._wiggle_state)
            return

        # Stop wiggle timer for non-wigglegram modes
        self._wiggle_timer.stop()
        self._wiggle_left_img = None
        self._wiggle_right_img = None

        if pil_img and self.preview_window.isVisible():
            pixmap = QtGui.QPixmap.fromImage(ImageQt(pil_img))
            self.preview_window.set_image(pixmap)
            self._save_sidecar_async()

    def _wiggle_step(self):
        """Toggle wiggle state and display the other frame."""
        self._wiggle_state = not self._wiggle_state
        self._show_wiggle_frame(self._wiggle_state)

    def _show_wiggle_frame(self, show_left: bool):
        """Display either left or right wigglegram frame."""
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

        # Handle case where frame_idx might be a string from combobox signal or other source
        try:
            frame_idx = int(float(frame_idx))
        except (ValueError, TypeError):
            frame_idx = self.ui.horizontalSlider.value()

        if frame_idx < 0:
            return
        params = self._get_render_params(frame_idx)
        if params:
            self.render_worker.render_frame(frame_idx, params)

    def toggle_preview_window(self):
        if self.preview_window.isVisible():
            self.preview_window.hide()
        else:
            screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
            size = self.preview_window.geometry()
            self.preview_window.move(screen.center().x() - size.width() // 2, screen.center().y() - size.height() // 2)
            self.preview_window.show()
            self._request_render()

    def browse_folder(self, line_edit):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

    def load_videos(self):
        src = self.ui.lineEdit_input_source.text()
        depth = self.ui.lineEdit_input_depth.text()
        sidecar = self.ui.lineEdit_inout_sidecar.text()
        multi = self.ui.checkBox_multi_map.isChecked()

        self.controller.set_sidecar_folder(sidecar)
        self.batch_controller.sidecar_folder = sidecar
        self.batch_controller.input_source_clips = src
        self.batch_controller.input_depth_maps = depth
        self.batch_controller.output_splatted = self.ui.lineEdit_output_splatted.text()
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
            self.ui.label_status.setText(f"Loaded {len(video_list)} videos.")

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

            self.ui.horizontalSlider.blockSignals(True)
            self.ui.horizontalSlider.setMaximum(total_frames - 1)
            self.ui.horizontalSlider.setValue(0)
            self.ui.horizontalSlider.blockSignals(False)

            self.ui.comboBox_map_select.clear()
            if self.ui.checkBox_multi_map.isChecked() and entry:
                map_folder = os.path.basename(os.path.dirname(entry["depth_map"]))
                self.ui.comboBox_map_select.addItem(map_folder)
            else:
                self.ui.comboBox_map_select.addItem("Default")

            self.ui.label_video_info.setText(f"Video: {new_idx + 1} / {len(self.controller.video_list)}")
            self.ui.lineEdit_jump_to.setText(str(new_idx + 1))
            self.ui.label_frame_info.setText(f"Frame: 1 / {total_frames}")
            self._request_render(0)
        else:
            logger.error(f"Could not load video index {new_idx}")

    def on_jump_to_clip(self):
        try:
            target = int(self.ui.lineEdit_jump_to.text())
            new_idx = max(0, min(target - 1, len(self.controller.video_list) - 1))
            self.current_video_idx = new_idx
            self.change_video(0)
        except ValueError:
            pass

    def on_frame_change(self, frame):
        total = self.controller.get_total_frames()
        self.ui.label_frame_info.setText(f"Frame: {frame + 1} / {total}")
        self._request_render(frame)

    def on_preview_scale_changed(self, value: str):
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

        current_mode = self.ui.comboBox_preview_source.currentText()

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
            idx = self.ui.comboBox_preview_source.findText(sidecar_mode)
            if idx >= 0:
                self.ui.comboBox_preview_source.setCurrentIndex(idx)

            self.ui.lineEdit_blur_bias.setText(str(sidecar_data.get("blur_bias", 0.5)))
            self.ui.lineEdit_mesh_extrusion.setText(str(sidecar_data.get("mesh_extrusion", 0.5)))
            self.ui.lineEdit_mesh_density.setText(str(sidecar_data.get("mesh_density", 0.5)))
            self.ui.lineEdit_mesh_dolly.setText(str(sidecar_data.get("mesh_dolly", 0.0)))

            steering = float(params.get("view_bias", 0.0))
            self.ui.lineEdit_mesh_bias.setText(f"{steering:.2f}")

            border_bias = float(params.get("border_bias", 0.0))
            self.ui.horizontalSlider_border_bias.setValue(int((border_bias * 50) + 50))
            self.ui.label_bias_value.setText(f"{border_bias:.2f}")

            self.ui.horizontalSlider_disparity.setValue(val("max_disp", 35))
            self.ui.horizontalSlider_convergence.setValue(int(float(params.get("convergence_point", 0.5)) * 100))
            self.ui.horizontalSlider_gamma.setValue(int(float(params.get("gamma", 1.0)) * 100))
            self.ui.horizontalSlider_dilate_x.setValue(val("dilate_x", 0))
            self.ui.horizontalSlider_dilate_y.setValue(val("dilate_y", 0))
            self.ui.horizontalSlider_blur_x.setValue(val("blur_x", 5))
            self.ui.horizontalSlider_blur_y.setValue(val("blur_y", 5))

            if "cross_view" in params:
                self.ui.checkBox_cross_view.setChecked(bool(params["cross_view"]))

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

        self.update_all_labels()
        logger.info(f"Sidecar loaded for video {self.current_video_idx + 1}")

    def load_sidecar(self):
        self._load_sidecar()

    def get_params(self) -> dict:
        try:
            preview_mode = self.ui.comboBox_preview_source.currentText()
            modes_without_cross_view = ["Original (Left Eye)", "Splat Result", "Splat Result(Low)"]

            logger.debug(f"get_params called, preview_mode: {preview_mode}")

            params = {
                "preview_source": preview_mode,
                "strict_ffmpeg_decode": self.ui.checkBox_ffmpeg.isChecked(),
                "max_disp": self.ui.horizontalSlider_disparity.value(),
                "convergence_point": self.ui.horizontalSlider_convergence.value() / 100.0,
                "gamma": self.ui.horizontalSlider_gamma.value() / 100.0,
                "border_mode": self.ui.comboBox_border.currentText(),
                "border_width": self.ui.horizontalSlider_border_width.value(),
                "view_bias": float(self.ui.lineEdit_mesh_bias.text() or 0.0),
                "border_bias": (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0,
                "cross_view": self.ui.checkBox_cross_view.isChecked()
                if preview_mode not in modes_without_cross_view
                else False,
                "dilate_x": self.ui.horizontalSlider_dilate_x.value(),
                "dilate_y": self.ui.horizontalSlider_dilate_y.value() / 2.0,
                "blur_x": self.ui.horizontalSlider_blur_x.value(),
                "blur_y": self.ui.horizontalSlider_blur_y.value(),
                "blur_bias": float(self.ui.lineEdit_blur_bias.text() or 0.5),
                "mesh_extrusion": float(self.ui.lineEdit_mesh_extrusion.text() or 0.5),
                "mesh_density": float(self.ui.lineEdit_mesh_density.text() or 0.5),
                "mesh_dolly": float(self.ui.lineEdit_mesh_dolly.text() or 0.0),
                "slider_disparity": self.ui.horizontalSlider_disparity.value(),
                "slider_convergence": self.ui.horizontalSlider_convergence.value(),
                "slider_bias": self.ui.horizontalSlider_border_bias.value(),
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
            input_source_clips=self.ui.lineEdit_input_source.text(),
            input_depth_maps=self.ui.lineEdit_input_depth.text(),
            output_splatted=self.ui.lineEdit_output_splatted.text(),
            max_disp=float(self.ui.horizontalSlider_disparity.value()),
            zero_disparity_anchor=float(self.ui.horizontalSlider_convergence.value()) / 100.0,
            depth_gamma=float(self.ui.horizontalSlider_gamma.value()) / 100.0,
            process_length=int(self.ui.lineEdit_process_length.text() or -1),
            enable_full_resolution=self.ui.checkBox_enable_full_res.isChecked(),
            full_res_batch_size=int(self.ui.lineEdit_high_batch.text() or 10),
            enable_low_resolution=self.ui.checkBox_enable_low_res.isChecked(),
            low_res_width=int(self.ui.lineEdit_low_width.text() or 1920),
            low_res_height=int(self.ui.lineEdit_low_height.text() or 1080),
            low_res_batch_size=int(self.ui.lineEdit_low_batch.text() or 50),
            dual_output=self.ui.checkBox_dual_output.isChecked(),
            strict_ffmpeg_decode=self.ui.checkBox_ffmpeg.isChecked(),
            move_to_finished=True,
            sidecar_folder=self.ui.lineEdit_inout_sidecar.text(),
            multi_map=self.ui.checkBox_multi_map.isChecked(),
            depth_dilate_size_x=float(self.ui.horizontalSlider_dilate_x.value()),
            depth_dilate_size_y=float(self.ui.horizontalSlider_dilate_y.value()),
            depth_blur_size_x=float(self.ui.horizontalSlider_blur_x.value()),
            depth_blur_size_y=float(self.ui.horizontalSlider_blur_y.value()),
        )
        return settings

    def _enable_inputs(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.ui.pushButton_start.setEnabled(enabled)
        self.ui.pushButton_single.setEnabled(enabled)
        self.ui.pushButton_stop.setEnabled(not enabled)
        self.ui.pushButton_load_refresh.setEnabled(enabled)

    def on_start_processing(self):
        self._enable_inputs(False)
        self.batch_controller.stop_event.clear()
        settings = self._get_processing_settings()
        self.batch_controller.start_batch(settings)
        self._processing_timer.start(100)

    def on_stop_processing(self):
        self.batch_controller.stop()
        self.ui.label_status.setText("Stopping...")

    def on_start_single_processing(self):
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
            self.ui.label_status.setText("Settings loaded.")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    def on_save_settings(self):
        config = self.get_params()
        self.save_config()
        self.ui.label_status.setText("Settings saved.")

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
            self.ui.label_status.setText(f"Saved settings to {os.path.basename(filename)}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def on_fsexport_to_sidecar(self):
        from core.splatting.fusion_export import FusionSidecarGenerator

        sidecar_path = self.controller.get_sidecar_path()
        if not sidecar_path:
            return
        generator = FusionSidecarGenerator()
        generator.generate_sidecar(sidecar_path, self.get_params())
        self.ui.label_status.setText("FSExport sidecar generated.")

    def on_update_from_sidecar(self):
        self._load_sidecar()
        self.ui.label_status.setText("Loaded sidecar data.")

    def on_auto_update_sidecar(self, checked):
        self._auto_save_sidecar = checked
        # Also use this to control auto-loading sidecar when video changes
        self.ui.action_auto_update_sidecar.setChecked(checked)

    def on_encoder_settings(self):
        from core.ui.qt_encoding_settings import QtEncodingSettingsDialog

        dialog = QtEncodingSettingsDialog(self, app_config={"encoding": self._encoding_config})
        if dialog.exec():
            self._encoding_config = dialog.get_config()
            self.save_config()
            self.ui.label_status.setText("Encoding settings saved")

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
        """Moves all files from 'finished' folders back to their original input folders."""
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

        source_clip_dir = self.ui.lineEdit_input_source.text()
        depth_map_dir = self.ui.lineEdit_input_depth.text()

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
        if reply == QtWidgets.QMessageBox.No:
            return

        source_clip_dir = self.ui.lineEdit_input_source.text()
        depth_map_dir = self.ui.lineEdit_input_depth.text()

        if not (os.path.isdir(source_clip_dir) and os.path.isdir(depth_map_dir)):
            QtWidgets.QMessageBox.warning(
                self,
                "Restore Error",
                "Restore 'finished' operation is only applicable when Input Source Clips and Input Depth Maps are set to directories (batch mode).",
            )
            return

        finished_source_folder = os.path.join(source_clip_dir, "finished")
        finished_depth_folder = os.path.join(depth_map_dir, "finished")

        restored_count = 0
        errors_count = 0

        if os.path.isdir(finished_source_folder):
            logger.info(f"Restoring source clips from: {finished_source_folder}")
            for filename in os.listdir(finished_source_folder):
                src_path = os.path.join(finished_source_folder, filename)
                dest_path = os.path.join(source_clip_dir, filename)
                if os.path.isfile(src_path):
                    try:
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                    except Exception as e:
                        errors_count += 1
                        logger.error(f"Error moving source clip '{filename}': {e}")

        if os.path.isdir(finished_depth_folder):
            logger.info(f"Restoring depth maps from: {finished_depth_folder}")
            for filename in os.listdir(finished_depth_folder):
                src_path = os.path.join(finished_depth_folder, filename)
                dest_path = os.path.join(depth_map_dir, filename)
                if os.path.isfile(src_path):
                    try:
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                    except Exception as e:
                        errors_count += 1
                        logger.error(f"Error moving depth map '{filename}': {e}")

        QtWidgets.QMessageBox.information(
            self,
            "Restore Complete",
            f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.",
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
                self.ui.label_status.setText(f"Loaded Fusion export: {filename}")
            else:
                QtWidgets.QMessageBox.warning(self, "Load Failed", "Could not load Fusion export file.")
        except Exception as e:
            logger.error(f"Failed to load fsexport: {e}")
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to load: {e}")

    def _apply_config_to_ui(self, config: dict):
        """Apply loaded config to UI widgets."""
        if "max_disp" in config:
            self.ui.horizontalSlider_disparity.setValue(int(config["max_disp"]))
        if "convergence_point" in config:
            self.ui.horizontalSlider_convergence.setValue(int(float(config["convergence_point"]) * 100))
        if "gamma" in config:
            self.ui.horizontalSlider_gamma.setValue(int(float(config["gamma"]) * 100))
        if "slider_border_width" in config:
            self.ui.horizontalSlider_border_width.setValue(config["slider_border_width"])
        if "slider_bias" in config:
            self.ui.horizontalSlider_border_bias.setValue(config["slider_bias"])
        if "input_source" in config:
            self.ui.lineEdit_input_source.setText(config["input_source"])
        if "input_depth" in config:
            self.ui.lineEdit_input_depth.setText(config["input_depth"])
        if "output_splatted" in config:
            self.ui.lineEdit_output_splatted.setText(config["output_splatted"])
        if "sidecar_path" in config:
            self.ui.lineEdit_inout_sidecar.setText(config["sidecar_path"])
        if "preview_source" in config:
            idx = self.ui.comboBox_preview_source.findText(config["preview_source"])
            if idx >= 0:
                self.ui.comboBox_preview_source.setCurrentIndex(idx)
        self.update_all_labels()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SplattingApp()
    window.show()
    sys.exit(app.exec())
