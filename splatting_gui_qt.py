import sys
import logging
import json
import os
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QDoubleValidator, QIntValidator
from core.ui.splatting_ui import Ui_MainWindow
from core.ui.preview_controller import PreviewController
from PIL.ImageQt import ImageQt


class PreviewWindow(QtWidgets.QDialog):
    """A simple pop-up window to show the rendered image."""

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

        # Player Setup
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)

        self.connect_signals()
        self.init_ui_defaults()
        # Restore previous session last to ensure it overwrites everything correctly
        self.load_config()

    def closeEvent(self, event):
        """Automatically save config when closing the window."""
        self.save_config()
        self.preview_window.close()
        event.accept()

    def save_config(self):
        """Saves current state to config_splat.splatcfg."""
        config = self.get_params()
        # Add paths and persistent fields
        config["input_source"] = self.ui.lineEdit_input_source.text()
        config["input_depth"] = self.ui.lineEdit_input_depth.text()
        config["output_splatted"] = self.ui.lineEdit_output_splatted.text()
        config["sidecar_path"] = self.ui.lineEdit_inout_sidecar.text()
        config["multi_map"] = self.ui.checkBox_multi_map.isChecked()
        config["low_width"] = str(self.ui.lineEdit_low_width.text())
        config["low_height"] = str(self.ui.lineEdit_low_height.text())
        config["process_length"] = str(self.ui.lineEdit_process_length.text())
        # Explicitly save slider POSITIONS (ints) so they load correctly
        config["slider_disparity"] = self.ui.horizontalSlider_disparity.value()
        config["slider_convergence"] = self.ui.horizontalSlider_convergence.value()
        config["slider_gamma"] = self.ui.horizontalSlider_gamma.value()
        config["slider_border_width"] = self.ui.horizontalSlider_border_width.value()
        config["slider_bias"] = self.ui.horizontalSlider_border_bias.value()

        try:
            with open("config_splat.splatcfg", "w") as f:
                json.dump(config, f, indent=4)
            print("Configuration saved.")
        except Exception as e:
            print(f"Error saving config: {e}")

    def load_config(self):
        """Restores state from config_splat.splatcfg."""
        if not os.path.exists("config_splat.splatcfg"):
            return

        try:
            with open("config_splat.splatcfg", "r") as f:
                config = json.load(f)

            # --- Restore Paths ---
            self.ui.lineEdit_input_source.setText(config.get("input_source", "./Clips"))
            self.ui.lineEdit_input_depth.setText(config.get("input_depth", "./Depth"))
            self.ui.lineEdit_output_splatted.setText(config.get("output_splatted", "./Splat"))
            self.ui.lineEdit_inout_sidecar.setText(str(config.get("sidecar_path", "./Sidecar")))
            self.ui.checkBox_multi_map.setChecked(config.get("multi_map", False))
            self.ui.lineEdit_low_width.setText(str(config.get("low_width", "640")))
            self.ui.lineEdit_low_height.setText(str(config.get("low_height", "320")))
            self.ui.lineEdit_process_length.setText(str(config.get("process_length", "0")))
            # --- Mesh ---
            self.ui.lineEdit_mesh_extrusion.setText(str(config.get("mesh_extrusion", "0.5")))
            self.ui.lineEdit_mesh_density.setText(str(config.get("mesh_density", "0.5")))
            self.ui.lineEdit_mesh_bias.setText(str(config.get("mesh_bias", "0.5")))
            self.ui.lineEdit_mesh_dolly.setText(str(config.get("mesh_dolly", "0.0")))

            # --- Restore Sliders ---
            self.ui.centralwidget.blockSignals(True)
            try:
                self.ui.horizontalSlider_disparity.setValue(config.get("slider_disparity", 35))
                self.ui.horizontalSlider_convergence.setValue(config.get("slider_convergence", 100))
                self.ui.horizontalSlider_gamma.setValue(config.get("slider_gamma", 99))
                self.ui.horizontalSlider_border_width.setValue(config.get("slider_border_width", 0))
                self.ui.horizontalSlider_border_bias.setValue(config.get("slider_bias", 50))
                self.ui.checkBox_cross_view.setChecked(config.get("cross_view", False))

                # Safety function for config loading
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

                # Restore checkboxes/combos
                idx = self.ui.comboBox_preview_source.findText(config.get("preview_source", "Splat Result"))
                if idx >= 0:
                    self.ui.comboBox_preview_source.setCurrentIndex(idx)

            finally:
                self.ui.centralwidget.blockSignals(False)

            self.update_all_labels()
            print("Configuration auto-loaded.")
        except Exception as e:
            # IMPORTANT: always make sure signals aren't stuck blocked
            self.ui.centralwidget.blockSignals(False)
            print(f"Failed to load config: {e}")

    def init_ui_defaults(self):
        """Fill dropdowns and set initial label values."""
        # Preview Source (Strictly matching PreviewRenderer.MODES)
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
        print(f"Initialized Preview Source with {len(modes)} modes.")
        # Border Modes
        self.ui.comboBox_border.clear()
        self.ui.comboBox_border.addItems(["Off", "Auto Basic", "Auto Adv.", "Manual"])
        # --- Numerical Validators for Premium UX ---
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

        # Initial label updates
        self.update_all_labels()

    def setup_slider(self, slider, label, divisor=1.0):
        """Helper to link a slider to a label and the preview update."""

        def on_change(value):
            if divisor == 0:
                label.setText(str(value))
            else:
                label.setText(f"{value / divisor:.2f}")
            self.update_preview()

        slider.valueChanged.connect(on_change)

    def update_all_labels(self):
        """Force labels to match slider values on startup."""
        self.ui.label_disparity_value.setText(str(self.ui.horizontalSlider_disparity.value()))
        self.ui.label_convergence_value.setText(f"{self.ui.horizontalSlider_convergence.value() / 100.0:.2f}")
        self.ui.label_gamma_value.setText(f"{self.ui.horizontalSlider_gamma.value() / 100.0:.2f}")
        self.ui.label_border_width_value.setText(str(self.ui.horizontalSlider_border_width.value()))
        # Bias is centered -1 to 1
        bias_val = (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0
        self.ui.label_bias_value.setText(f"{bias_val:.2f}")

    def connect_signals(self):
        # --- Browsing ---
        self.ui.pushButton_browse_source.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_input_source))
        self.ui.pushButton_browse_depth.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_input_depth))
        self.ui.pushButton_browse_output.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_output_splatted))
        self.ui.pushButton_browse_sidecar.clicked.connect(lambda: self.browse_folder(self.ui.lineEdit_inout_sidecar))
        self.ui.pushButton_load_refresh.clicked.connect(self.load_videos)

        # --- Navigation ---
        self.ui.horizontalSlider.valueChanged.connect(self.on_frame_change)
        self.ui.pushButton_next.clicked.connect(lambda: self.change_video(1))
        self.ui.pushButton_prev.clicked.connect(lambda: self.change_video(-1))

        # --- Playback ---
        self.ui.pushButton_play.clicked.connect(self.toggle_playback)
        self.ui.pushButton_fast_forward.clicked.connect(lambda: self.toggle_playback(fast=True))
        # Update playback speed in real-time
        self.ui.spinBox_ff_speed.valueChanged.connect(lambda: self.update_playback_speed())

        # --- Jump To Clip ---
        self.ui.lineEdit_jump_to.returnPressed.connect(self.on_jump_to_clip)

        # --- Settings Toggle ---
        self.ui.pushButton_settings.clicked.connect(self.toggle_preview_window)

        # --- Param Wiring with Value Labels ---
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

        # Custom Wiring for Bias (-1 to 1 centered)
        self.ui.horizontalSlider_border_bias.valueChanged.connect(self.on_bias_change)

        # Combo boxes too
        self.ui.comboBox_preview_source.currentIndexChanged.connect(self.update_preview)
        self.ui.comboBox_border.currentIndexChanged.connect(self.update_preview)

        # Line Edits (Mesh/Blur)
        self.ui.lineEdit_blur_bias.textChanged.connect(self.update_preview)
        self.ui.lineEdit_mesh_extrusion.textChanged.connect(self.update_preview)
        self.ui.lineEdit_mesh_density.textChanged.connect(self.update_preview)
        self.ui.lineEdit_mesh_bias.textChanged.connect(self.update_preview)

    def on_bias_change(self, value):
        """Centered Border Bias 0..100 -> -1..1 logic."""
        bias_val = (value - 50) / 50.0
        self.ui.label_bias_value.setText(f"{bias_val:.2f}")
        self.update_preview()

    def toggle_playback(self, fast=False):
        """Starts or stops the frame timer."""
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.ui.pushButton_play.setText("▶")
        else:
            self.ui.pushButton_play.setText("⏸")
            # --- Speed Logic ---
            # Play = 24 FPS (41ms)
            # FF = Faster + Skipping frames
            self.is_fast_forward = fast
            interval = 41 if not fast else 30
            self.play_timer.start(interval)

    def update_playback_speed(self, is_fast=None):
        """Helper to react to spinbox changes while playing."""
        if not self.play_timer.isActive():
            return

        # 80ms is roughly 12.5 FPS (decent for preview)
        interval = 80
        if is_fast or (is_fast is None and self.ui.pushButton_play.text() == "⏸"):
            # Get step speed
            speed = max(1, self.ui.spinBox_ff_speed.value())
            # If speed is high (>10), we actually just want a faster interval too
            interval = max(30, 1000 // (speed + 10))

        self.play_timer.setInterval(interval)

    def advance_frame(self):
        """Moves the slider forward. Uses spinbox for 'skip' steps if fast."""
        curr = self.ui.horizontalSlider.value()
        max_val = self.ui.horizontalSlider.maximum()

        # Get step: FF uses spinbox, Regular Play uses +1
        step = self.ui.spinBox_ff_speed.value() if getattr(self, "is_fast_forward", False) else 1

        new_frame = curr + step
        if new_frame > max_val:
            self.ui.horizontalSlider.setValue(0)
        else:
            self.ui.horizontalSlider.setValue(new_frame)

    def toggle_preview_window(self):
        if self.preview_window.isVisible():
            self.preview_window.hide()
        else:
            # Center on screen
            screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
            size = self.preview_window.geometry()
            self.preview_window.move(screen.center().x() - size.width() // 2, screen.center().y() - size.height() // 2)
            self.preview_window.show()
            self.update_preview()

    def browse_folder(self, line_edit):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line_edit.setText(folder)

    def load_videos(self):
        src = self.ui.lineEdit_input_source.text()
        depth = self.ui.lineEdit_input_depth.text()
        multi = self.ui.checkBox_multi_map.isChecked()

        # Capture current video path to try and restore it later
        old_path = None
        if self.controller.video_list:
            old_path = self.controller.video_list[self.current_video_idx]["source_video"]

        video_list = self.controller.load_video_list(src, depth, multi)
        if video_list:
            # Try to find the old video in the new list
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

            # --- SIDECAR AUTO-LOAD ---
            self.load_sidecar()

            # Update Video Info
            entry = self.controller.video_list[new_idx]
            print(f"Loading: {entry['source_video']}")

            # Update Sliders
            self.ui.horizontalSlider.setMaximum(self.controller.total_frames - 1)
            self.ui.horizontalSlider.setValue(0)

            # --- Update ComboBox (Multi-Map) ---
            self.ui.comboBox_map_select.clear()
            if self.ui.checkBox_multi_map.isChecked():
                # Extract folder name: Z:/Path/Depth_Midas/frame_0001.png -> Depth_Midas
                map_folder = os.path.basename(os.path.dirname(entry["depth_map"]))
                self.ui.comboBox_map_select.addItem(map_folder)
            else:
                self.ui.comboBox_map_select.addItem("Default")

            # Update Info Labels
            self.ui.label_video_info.setText(f"Video: {new_idx + 1} / {len(self.controller.video_list)}")
            self.ui.lineEdit_jump_to.setText(str(new_idx + 1))
            self.update_preview()
        else:
            print(f"Error: Could not load video index {new_idx}")

    def on_jump_to_clip(self):
        """When user types a number in 'Jump to' and hits Enter, navigate to that video."""
        try:
            target = int(self.ui.lineEdit_jump_to.text())
            # Convert 1-indexed UI to 0-indexed code
            new_idx = max(0, min(target - 1, len(self.controller.video_list) - 1))
            self.current_video_idx = new_idx
            self.change_video(0)
        except ValueError:
            pass

    def on_frame_change(self, frame):
        # Update Label
        self.ui.label_frame_info.setText(f"Frame: {frame + 1} / {self.controller.total_frames}")
        self.update_preview()

    def update_preview(self):
        frame = self.ui.horizontalSlider.value()
        params = self.get_params()

        # Debug the current mode
        if params:
            # For Wigglegram, we alternate the 'wiggle_toggle' based on the current frame or time
            params["wiggle_toggle"] = frame % 2 == 0
            # Ensure mode is explicitly in params
            params["mode"] = params.get("preview_source", "Splat Result")

        pil_img = self.controller.get_frame(frame, params)

        if pil_img:
            if self.preview_window.isVisible():
                pixmap = QtGui.QPixmap.fromImage(ImageQt(pil_img))
                self.preview_window.set_image(pixmap)

            # AUTO-SAVE SIDECAR (Only if you've moved a slider)
            self.save_sidecar()
        else:
            if self.controller.source_reader:
                print(f"Warning: No image returned for frame {frame}")

    def get_sidecar_path(self):
        """Builds the path to the sidecar file for the current video."""
        if not self.controller.video_list:
            return None
        entry = self.controller.video_list[self.current_video_idx]

        # We usually store sidecars in the sidecar folder named after the source video
        filename = os.path.basename(entry["source_video"])
        sidecar_filename = os.path.splitext(filename)[0] + ".json"

        sidecar_root = self.ui.lineEdit_inout_sidecar.text()
        if not os.path.exists(sidecar_root):
            os.makedirs(sidecar_root, exist_ok=True)

        return os.path.join(sidecar_root, sidecar_filename)

    def save_sidecar(self):
        """Saves current settings for THIS specific video."""
        path = self.get_sidecar_path()
        if not path:
            return

        params = self.get_params()
        try:
            with open(path, "w") as f:
                json.dump(params, f, indent=4)
        except Exception as e:
            print(f"Sidecar save failed: {e}")

    def load_sidecar(self):
        """Loads settings for THIS specific video if they exist."""
        path = self.get_sidecar_path()

        # Save current UI mode before sidecar possible override
        current_mode = self.ui.comboBox_preview_source.currentText()

        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Apply to UI
            self.ui.centralwidget.blockSignals(True)
            try:
                # Helper for loading
                def val(key, default):
                    v = data.get(key, default)
                    return int(float(v)) if v is not None else int(default)

                # Block all specific signals
                self.ui.horizontalSlider_disparity.blockSignals(True)
                self.ui.horizontalSlider_convergence.blockSignals(True)
                self.ui.horizontalSlider_gamma.blockSignals(True)
                self.ui.horizontalSlider_border_bias.blockSignals(True)
                self.ui.horizontalSlider_dilate_x.blockSignals(True)
                self.ui.horizontalSlider_dilate_y.blockSignals(True)
                self.ui.horizontalSlider_blur_x.blockSignals(True)
                self.ui.horizontalSlider_blur_y.blockSignals(True)

                # 1. Restore Paths & Mode
                sidecar_mode = data.get("preview_source", current_mode)
                idx = self.ui.comboBox_preview_source.findText(sidecar_mode)
                if idx >= 0:
                    self.ui.comboBox_preview_source.setCurrentIndex(idx)

                # 2. Restore Text Settings
                self.ui.lineEdit_blur_bias.setText(str(data.get("blur_bias", 0.5)))
                self.ui.lineEdit_mesh_extrusion.setText(str(data.get("mesh_extrusion", 1.0)))
                self.ui.lineEdit_mesh_density.setText(str(data.get("mesh_density", 1.0)))
                self.ui.lineEdit_mesh_dolly.setText(str(data.get("mesh_dolly", 0.0)))

                # 3. Restore Steiner (Bias) logic
                # Steering is independent now
                steering = float(data.get("view_bias", 0.0))
                self.ui.lineEdit_mesh_bias.setText(f"{steering:.2f}")

                # Border Bias slider
                border_bias = float(data.get("border_bias", 0.0))
                self.ui.horizontalSlider_border_bias.setValue(int((border_bias * 50) + 50))
                self.ui.label_bias_value.setText(f"{border_bias:.2f}")

                # 4. Restore Base Sliders
                self.ui.horizontalSlider_disparity.setValue(val("max_disp", 35))
                self.ui.horizontalSlider_convergence.setValue(int(float(data.get("convergence_point", 0.5)) * 100))
                self.ui.horizontalSlider_gamma.setValue(int(float(data.get("gamma", 1.0)) * 100))
                self.ui.horizontalSlider_dilate_x.setValue(val("dilate_x", 0))
                self.ui.horizontalSlider_dilate_y.setValue(val("dilate_y", 0))
                self.ui.horizontalSlider_blur_x.setValue(val("blur_x", 5))
                self.ui.horizontalSlider_blur_y.setValue(val("blur_y", 5))

                # 5. Checkboxes
                if "cross_view" in data:
                    self.ui.checkBox_cross_view.setChecked(bool(data["cross_view"]))

            finally:
                # Unblock EVERYTHING
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
            print(f"Sidecar loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"Sidecar load failed for {path}: {e}")
            self.ui.centralwidget.blockSignals(False)

    def get_params(self) -> dict:
        """The 'Source of Truth' mapping. Everything must match core/splatting/ keys."""
        try:
            return {
                # --- Rendering Mode ---
                "preview_source": self.ui.comboBox_preview_source.currentText(),
                "strict_ffmpeg_decode": self.ui.checkBox_ffmpeg.isChecked(),
                # --- Stereo Engine Keys ---
                "max_disp": self.ui.horizontalSlider_disparity.value(),
                "convergence_point": self.ui.horizontalSlider_convergence.value() / 100.0,
                "gamma": self.ui.horizontalSlider_gamma.value() / 100.0,
                # --- Border Engine Keys ---
                "border_mode": self.ui.comboBox_border.currentText(),
                "border_width": self.ui.horizontalSlider_border_width.value(),
                # THIS is the Camera steering bias (-1 to 1)
                "view_bias": float(self.ui.lineEdit_mesh_bias.text() or 0.0),
                # THIS is the Splatting border bias (-1 to 1)
                "border_bias": (self.ui.horizontalSlider_border_bias.value() - 50) / 50.0,
                # --- Cross View Toggle ---
                "cross_view": self.ui.checkBox_cross_view.isChecked(),
                # --- Depth Pre-processing Engine Keys ---
                "dilate_x": self.ui.horizontalSlider_dilate_x.value(),
                # Note: / 2.0 logic from old GUI
                "dilate_y": self.ui.horizontalSlider_dilate_y.value() / 2.0,
                "blur_x": self.ui.horizontalSlider_blur_x.value(),
                "blur_y": self.ui.horizontalSlider_blur_y.value(),
                # Correctly mapping Blur L bias to its own LineEdit
                "blur_bias": float(self.ui.lineEdit_blur_bias.text() or 0.5),
                "mesh_extrusion": float(self.ui.lineEdit_mesh_extrusion.text() or 1.0),
                "mesh_density": float(self.ui.lineEdit_mesh_density.text() or 1.0),
                "mesh_dolly": float(self.ui.lineEdit_mesh_dolly.text() or 0.0),
                # Record raw slider values for Sidecar persistence
                "slider_disparity": self.ui.horizontalSlider_disparity.value(),
                "slider_convergence": self.ui.horizontalSlider_convergence.value(),
                "slider_bias": self.ui.horizontalSlider_border_bias.value(),
            }
        except Exception:
            # Fallback for when line edits are empty while typing
            return {}


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SplattingApp()
    window.show()
    sys.exit(app.exec())
