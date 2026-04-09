from PySide6 import QtWidgets, QtCore
from core.ui.encode_ui import Ui_Dialog
from core.common.encoding_utils import (
    QUALITY_PRESETS,
    CPU_TUNE_OPTIONS,
    ENCODER_OPTIONS as ENCODER_OPTS_DICT,
    CODEC_OPTIONS,
    CONTAINER_OPTIONS as CONTAINER_OPTS_DICT,
    DEFAULT_ENCODING_CONFIG,
    build_encoder_args,
)
from core.common import gpu_utils


class QtEncodingSettingsDialog(QtWidgets.QDialog):
    """Qt-based encoding settings dialog using encode_ui."""

    def __init__(self, parent=None, app_config=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self._app_config = app_config or {}
        self._init_options()
        self._load_config()
        self._connect_signals()
        self._connect_dynamic_signals()

    def _init_options(self):
        self.ui.comboBox_codec.addItems(CODEC_OPTIONS)
        self._update_container_options()
        self._update_encoder_options()

        self.ui.comboBox_quality.addItems(list(QUALITY_PRESETS))
        self.ui.comboBox_cpu_tune.addItems(CPU_TUNE_OPTIONS)

        self.ui.comboBox_color_tag.addItems(["Off", "Auto", "BT.709 L", "BT.709 F", "BT.2020 PQ", "BT.2020 HLG"])

    def _connect_dynamic_signals(self):
        self.ui.comboBox_codec.currentTextChanged.connect(self._on_codec_changed)

    def _on_codec_changed(self, codec: str):
        self._update_container_options()
        self._update_encoder_options()

    def _update_container_options(self):
        codec = self.ui.comboBox_codec.currentText()
        containers = CONTAINER_OPTS_DICT.get(codec, ("MP4",))
        self.ui.comboBox_container.blockSignals(True)
        self.ui.comboBox_container.clear()
        self.ui.comboBox_container.addItems(containers)
        self.ui.comboBox_container.blockSignals(False)

    def _update_encoder_options(self):
        codec = self.ui.comboBox_codec.currentText()
        encoders = ENCODER_OPTS_DICT.get(codec, ("Auto", "Force CPU"))
        self.ui.comboBox_encoder.blockSignals(True)
        self.ui.comboBox_encoder.clear()
        self.ui.comboBox_encoder.addItems(encoders)
        self.ui.comboBox_encoder.blockSignals(False)

    def _load_config(self):
        config = self._app_config.get("encoding", DEFAULT_ENCODING_CONFIG)
        self.ui.comboBox_codec.setCurrentText(config.get("codec", "H.264"))
        self._update_container_options()
        self._update_encoder_options()
        self.ui.comboBox_container.setCurrentText(config.get("container", "MP4"))
        self.ui.comboBox_encoder.setCurrentText(config.get("encoder", "Auto"))
        self.ui.comboBox_quality.setCurrentText(config.get("quality", "Medium"))
        self.ui.comboBox_cpu_tune.setCurrentText(config.get("cpu_tune", "default"))
        self.ui.comboBox_color_tag.setCurrentText(config.get("color_tag", "Auto"))

        self.ui.lineEdit_full_res.setText(str(config.get("crf_high", 18)))
        self.ui.lineEdit_low_res.setText(str(config.get("crf_low", 23)))

        self.ui.checkBox_lookahead.setChecked(config.get("nvenc_lookahead", False))
        if hasattr(self.ui, "comboBox_look_ahead_frames"):
            self.ui.comboBox_look_ahead_frames.setCurrentText(str(config.get("nvenc_lookahead_frames", 10)))
        self.ui.checkBox_spatial_aq.setChecked(config.get("nvenc_spatial_aq", True))
        self.ui.checkBox_temporal_aq.setChecked(config.get("nvenc_temporal_aq", True))
        if hasattr(self.ui, "comboBox_strength_aq"):
            self.ui.comboBox_strength_aq.setCurrentText(str(config.get("nvenc_aq_strength", 5)))

        self.ui.checkBox.setChecked(config.get("output_mask_low", True))
        self.ui.checkBox_flowmap_full.setChecked(config.get("output_flowmap_full", False))
        self.ui.checkBox_mask.setChecked(config.get("output_mask_high", True))
        self.ui.checkBox_splat_sbs.setChecked(config.get("output_splat_sbs", True))

    def _connect_signals(self):
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)

        # Connect lookahead checkbox to frames combo
        self.ui.checkBox_lookahead.stateChanged.connect(self._on_lookahead_toggled)

    def _on_lookahead_toggled(self, state):
        enabled = state != 0
        if hasattr(self.ui, "comboBox_look_ahead_frames"):
            self.ui.comboBox_look_ahead_frames.setEnabled(enabled)

    def get_config(self) -> dict:
        def get_combo_text(combo_box):
            if combo_box and hasattr(combo_box, "currentText"):
                return combo_box.currentText()
            return "10"

        def get_combo_int(combo_box, default=5):
            if combo_box and hasattr(combo_box, "currentText"):
                try:
                    return int(combo_box.currentText())
                except:
                    return default
            return default

        return {
            "codec": self.ui.comboBox_codec.currentText(),
            "container": self.ui.comboBox_container.currentText(),
            "encoder": self.ui.comboBox_encoder.currentText(),
            "quality": self.ui.comboBox_quality.currentText(),
            "cpu_tune": self.ui.comboBox_cpu_tune.currentText(),
            "color_tag": self.ui.comboBox_color_tag.currentText(),
            "crf_high": int(self.ui.lineEdit_full_res.text() or 18),
            "crf_low": int(self.ui.lineEdit_low_res.text() or 23),
            "nvenc_lookahead": self.ui.checkBox_lookahead.isChecked(),
            "nvenc_lookahead_frames": get_combo_int(getattr(self.ui, "comboBox_look_ahead_frames", None), 10),
            "nvenc_spatial_aq": self.ui.checkBox_spatial_aq.isChecked(),
            "nvenc_temporal_aq": self.ui.checkBox_temporal_aq.isChecked(),
            "nvenc_aq_strength": get_combo_int(getattr(self.ui, "comboBox_strength_aq", None), 5),
            "output_mask_low": self.ui.checkBox.isChecked(),
            "output_flowmap_full": self.ui.checkBox_flowmap_full.isChecked(),
            "output_mask_high": self.ui.checkBox_mask.isChecked(),
            "output_splat_sbs": self.ui.checkBox_splat_sbs.isChecked(),
        }

    def get_encoder_args(self, is_low_res=False) -> list:
        config = self.get_config()
        crf = config["crf_low"] if is_low_res else config["crf_high"]
        return build_encoder_args(
            codec=config["codec"],
            container=config["container"],
            encoder=config["encoder"],
            quality=config["quality"],
            crf=crf,
            cpu_tune=config["cpu_tune"],
            color_tag=config["color_tag"],
            nvenc_lookahead=config["nvenc_lookahead"],
            nvenc_lookahead_frames=config["nvenc_lookahead_frames"],
            nvenc_spatial_aq=config["nvenc_spatial_aq"],
            nvenc_temporal_aq=config["nvenc_temporal_aq"],
            nvenc_aq_strength=config["nvenc_aq_strength"],
        )
