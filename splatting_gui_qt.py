import sys
import logging
import json
import os
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QDoubleValidator, QIntValidator
from core.ui.splatting_ui import Ui_MainWindow
from core.ui.preview_controller import PreviewController
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
        self.layout.addWidget(self.label)
        self.resize(800, 600)

    def set_image(self, pixmap):
        self.label.setPixmap(
            pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )


class SplattingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.controller = PreviewController()
        self.preview_window = PreviewWindow(self)

        self.current_video_idx = 0

        self._setup_workers()
        self._setup_playback_timer()
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
        self.playback_worker.playback_started.connect(lambda: self.ui.pushButton_play.setText("⏸"))
        self.playback_worker.playback_stopped.connect(lambda: self.ui.pushButton_play.setText("▶"))

    def _setup_playback_timer(self):
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.playback_worker.tick)
        self._is_fast_forward = False

    def closeEvent(self, event):
        self.save_config()
        self.play_timer.stop()
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

            self.ui.lineEdit_input_source.setText(config.get("input_source", "./Clips"))
            self.ui.lineEdit_input_depth.setText(config.get("input_depth", "./Depth"))
            self.ui.lineEdit_output_splatted.setText(config.get("output_splatted", "./Splat"))
            self.ui.lineEdit_inout_sidecar.setText(str(config.get("sidecar_path", "./Sidecar")))
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
            "Original (Left Eye)",
            "Depth Map",
            "Depth Map (Color)",
            "Anaglyph 3D",
            "Dubois Anaglyph",
            "Optimized Anaglyph",
            "Wigglegram",
            "Side-by-Side",
            "Mesh Warp",
            "SBS + Mesh",
            "Mesh Warp SBS",
        ]
        self.ui.comboBox_preview_source.addItems(modes)
        logger.info(f"Initialized Preview Source with {len(modes)} modes.")

        self.ui.comboBox_border.clear()
        self.ui.comboBox_border.addItems(["Off", "Auto Basic", "Auto Adv.", "Manual"])

        self.ui.lineEdit_mesh_extrusion.setValidator(QDoubleValidator(-1.0, 5.0, 3, self))
        self.ui.lineEdit_mesh_density.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
        self.ui.lineEdit_mesh_bias.setValidator(QDoubleValidator(-1.0, 1.0, 3, self))
        self.ui.lineEdit_mesh_dolly.setValidator(QDoubleValidator(0.0, 100.0, 3, self))

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

        self.ui.lineEdit_jump_to.returnPressed.connect(self.on_jump_to_clip)

        self.ui.pushButton_settings.clicked.connect(self.toggle_preview_window)

        mappings = [
            (self.ui.horizontalSlider_disparity, self.ui.label_disparity_value, 1.0),
            (self.ui.horizontalSlider_convergence, self.ui.label_convergence_value, 100.0),
            (self.ui.horizontalSlider_gamma, self.ui.label_gamma_value, 100.0),
            (self.ui.horizontalSlider_border_width, self.ui.label_border_width_value, 1.0),
            (self.ui.horizontalSlider_dilate_x, self.ui.label_dilate_x_value, 1.0),
            (self.ui.horizontalSlider_dilate_y, self.ui.label_dilate_y_value, 1.0),
            (self.ui.horizontalSlider_blur_x, self.ui.label_blue_x_value, 1.0),
            (self.ui.horizontalSlider_blur_y, self.ui.label_blur_y_value, 1.0),
        ]

        for slider, label, div in mappings:
            self.setup_slider(slider, label, div)

        self.ui.horizontalSlider_border_bias.valueChanged.connect(self.on_bias_change)

        self.ui.comboBox_preview_source.currentIndexChanged.connect(self._request_render)
        self.ui.comboBox_border.currentIndexChanged.connect(self._request_render)

        self.ui.lineEdit_blur_bias.textChanged.connect(self._request_render)
        self.ui.lineEdit_mesh_extrusion.textChanged.connect(self._request_render)
        self.ui.lineEdit_mesh_density.textChanged.connect(self._request_render)
        self.ui.lineEdit_mesh_bias.textChanged.connect(self._request_render)

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
        self.ui.pushButton_play.setText("▶")

    def _on_frame_ready(self, pil_img):
        if pil_img and self.preview_window.isVisible():
            pixmap = QtGui.QPixmap.fromImage(ImageQt(pil_img))
            self.preview_window.set_image(pixmap)
        self._save_sidecar_async()

    def _on_render_error(self, error_msg):
        logger.error(f"Render error: {error_msg}")

    def _request_render(self, frame_idx=None):
        if not self.controller.video_list:
            return
        if frame_idx is None:
            frame_idx = self.ui.horizontalSlider.value()
        frame_idx = int(float(frame_idx))
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
        multi = self.ui.checkBox_multi_map.isChecked()

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

    def change_video(self, delta):
        new_idx = self.current_video_idx + delta
        if self.controller.set_current_video(new_idx, self.get_params()):
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
        params = self.get_params()
        self.controller.save_sidecar(params)

    def save_sidecar(self):
        params = self.get_params()
        self.controller.save_sidecar(params)

    def _load_sidecar(self):
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
            self.ui.lineEdit_mesh_extrusion.setText(str(sidecar_data.get("mesh_extrusion", 1.0)))
            self.ui.lineEdit_mesh_density.setText(str(sidecar_data.get("mesh_density", 1.0)))
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
            return {
                "preview_source": self.ui.comboBox_preview_source.currentText(),
                "strict_ffmpeg_decode": self.ui.checkBox_ffmpeg.isChecked(),
                "max_disp": self.ui.horizontalSlider_disparity.value(),
                "convergence_point": self.ui.horizontalSlider_convergence.value() / 100.0,
                "gamma": self.ui.horizontalSlider_gamma.value() / 100.0,
                "border_mode": self.ui.comboBox_border.currentText(),
                "border_width": self.ui.horizontalSlider_border_width.value(),
                "view_bias": float(self.ui.lineEdit_mesh_bias.text() or 0.0),
                "border_bias": (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0,
                "cross_view": self.ui.checkBox_cross_view.isChecked(),
                "dilate_x": self.ui.horizontalSlider_dilate_x.value(),
                "dilate_y": self.ui.horizontalSlider_dilate_y.value() / 2.0,
                "blur_x": self.ui.horizontalSlider_blur_x.value(),
                "blur_y": self.ui.horizontalSlider_blur_y.value(),
                "blur_bias": float(self.ui.lineEdit_blur_bias.text() or 0.5),
                "mesh_extrusion": float(self.ui.lineEdit_mesh_extrusion.text() or 1.0),
                "mesh_density": float(self.ui.lineEdit_mesh_density.text() or 1.0),
                "mesh_dolly": float(self.ui.lineEdit_mesh_dolly.text() or 0.0),
                "slider_disparity": self.ui.horizontalSlider_disparity.value(),
                "slider_convergence": self.ui.horizontalSlider_convergence.value(),
                "slider_bias": self.ui.horizontalSlider_border_bias.value(),
            }
        except Exception:
            return {}


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SplattingApp()
    window.show()
    sys.exit(app.exec())
