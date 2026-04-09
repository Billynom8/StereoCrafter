# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'splatting.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QSpinBox, QStatusBar, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1309, 755)
        icon = QIcon()
        icon.addFile(u"core/ui/icons/logo.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet(u"\n"
"/* Only set font, let colors be native */\n"
"QWidget { \n"
"    font-family: 'Segoe UI', sans-serif; \n"
"    font-size: 9pt; \n"
"}\n"
"\n"
"/* Make Group Boxes look clean without hard-coding background colors */\n"
"QGroupBox { \n"
"    font-weight: bold; \n"
"    border: 1px solid palette(mid); /* Uses system border color */\n"
"    border-radius: 5px; \n"
"    margin-top: 1ex; \n"
"    padding-top: 10px; \n"
"    background-color: transparent; /* Let system decide bg */\n"
"}\n"
"QGroupBox::title { \n"
"    subcontrol-origin: margin; \n"
"    left: 10px; \n"
"    padding: 0 3px 0 3px; \n"
"}\n"
"\n"
"/* CUSTOM SLIDER STYLING (This fixes the ugly orange dot) */\n"
"QSlider::groove:horizontal { \n"
"    border: 1px solid palette(dark); \n"
"    height: 4px; \n"
"    background: palette(mid);\n"
"    margin: 2px 0; \n"
"    border-radius: 2px; \n"
"}\n"
"QSlider::handle:horizontal { \n"
"    background: palette(button);\n"
"    border: 1px solid palette(mid); \n"
"    width: 10px; \n"
"    height: 10px;"
                        "\n"
"    margin: -3px 0; \n"
"    border-radius: 2px; \n"
"}\n"
"QSlider::handle:horizontal:hover {\n"
"    background: palette(highlight);\n"
"}\n"
"\n"
"/* Keep inputs readable */\n"
"QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { \n"
"    background-color: palette(base); \n"
"    border: 1px solid palette(mid); \n"
"    color: palette(text); \n"
"    padding: 2px; \n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"/* Keep buttons looking like buttons */\n"
"QPushButton { \n"
"    background-color: palette(button); \n"
"    border: 1px solid palette(mid); \n"
"    color: palette(button-text); \n"
"    padding: 4px 10px; \n"
"    border-radius: 3px; \n"
"}\n"
"QPushButton:hover { \n"
"    background-color: palette(light); \n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: palette(dark); \n"
"}\n"
"QPushButton:checked {\n"
"    background-color: palette(highlight);\n"
"    color: palette(highlighted-text);\n"
"    border: 1px solid palette(highlight);\n"
"    font-weight: bold;\n"
"}\n"
"QPushButton:checke"
                        "d:hover {\n"
"    background-color: palette(light);\n"
"}\n"
"/* Progress bar */\n"
"QProgressBar { \n"
"    border: 1px solid palette(mid); \n"
"    border-radius: 3px; \n"
"    text-align: center; \n"
"    background-color: palette(window); \n"
"}\n"
"QProgressBar::chunk { \n"
"    background-color: palette(highlight); \n"
"    width: 1px; \n"
"}\n"
"   ")
        self.action_load_settings = QAction(MainWindow)
        self.action_load_settings.setObjectName(u"action_load_settings")
        self.action_save = QAction(MainWindow)
        self.action_save.setObjectName(u"action_save")
        self.action_save_to_file = QAction(MainWindow)
        self.action_save_to_file.setObjectName(u"action_save_to_file")
        self.action_load_fsexport = QAction(MainWindow)
        self.action_load_fsexport.setObjectName(u"action_load_fsexport")
        self.action_fsexport_to_sidecar = QAction(MainWindow)
        self.action_fsexport_to_sidecar.setObjectName(u"action_fsexport_to_sidecar")
        self.action_restore_from_finished = QAction(MainWindow)
        self.action_restore_from_finished.setObjectName(u"action_restore_from_finished")
        self.action_encoder = QAction(MainWindow)
        self.action_encoder.setObjectName(u"action_encoder")
        self.action_update_from_sidecar = QAction(MainWindow)
        self.action_update_from_sidecar.setObjectName(u"action_update_from_sidecar")
        self.action_update_from_sidecar.setCheckable(True)
        self.action_auto_update_sidecar = QAction(MainWindow)
        self.action_auto_update_sidecar.setObjectName(u"action_auto_update_sidecar")
        self.action_auto_update_sidecar.setCheckable(True)
        self.action_guide = QAction(MainWindow)
        self.action_guide.setObjectName(u"action_guide")
        self.action_calculator = QAction(MainWindow)
        self.action_calculator.setObjectName(u"action_calculator")
        self.action_debug = QAction(MainWindow)
        self.action_debug.setObjectName(u"action_debug")
        self.action_debug.setCheckable(True)
        self.action_about = QAction(MainWindow)
        self.action_about.setObjectName(u"action_about")
        self.action_exit = QAction(MainWindow)
        self.action_exit.setObjectName(u"action_exit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout_1 = QGridLayout()
        self.gridLayout_1.setObjectName(u"gridLayout_1")
        self.groupBox_input_output = QGroupBox(self.centralwidget)
        self.groupBox_input_output.setObjectName(u"groupBox_input_output")
        self.groupBox_input_output.setMaximumSize(QSize(16777215, 160))
        self.gridLayout_2 = QGridLayout(self.groupBox_input_output)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.pushButton_browse_output = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_output.setObjectName(u"pushButton_browse_output")

        self.gridLayout_2.addWidget(self.pushButton_browse_output, 2, 2, 1, 1)

        self.label_input_depth = QLabel(self.groupBox_input_output)
        self.label_input_depth.setObjectName(u"label_input_depth")
        self.label_input_depth.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_input_depth, 1, 0, 1, 1)

        self.pushButton_browse_depth = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_depth.setObjectName(u"pushButton_browse_depth")

        self.gridLayout_2.addWidget(self.pushButton_browse_depth, 1, 2, 1, 1)

        self.lineEdit_input_source = QLineEdit(self.groupBox_input_output)
        self.lineEdit_input_source.setObjectName(u"lineEdit_input_source")

        self.gridLayout_2.addWidget(self.lineEdit_input_source, 0, 1, 1, 1)

        self.pushButton_select_depth = QPushButton(self.groupBox_input_output)
        self.pushButton_select_depth.setObjectName(u"pushButton_select_depth")

        self.gridLayout_2.addWidget(self.pushButton_select_depth, 1, 3, 1, 1)

        self.checkBox_multi_map = QCheckBox(self.groupBox_input_output)
        self.checkBox_multi_map.setObjectName(u"checkBox_multi_map")

        self.gridLayout_2.addWidget(self.checkBox_multi_map, 2, 3, 1, 1)

        self.lineEdit_output_splatted = QLineEdit(self.groupBox_input_output)
        self.lineEdit_output_splatted.setObjectName(u"lineEdit_output_splatted")

        self.gridLayout_2.addWidget(self.lineEdit_output_splatted, 2, 1, 1, 1)

        self.label_input_source = QLabel(self.groupBox_input_output)
        self.label_input_source.setObjectName(u"label_input_source")
        self.label_input_source.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_input_source, 0, 0, 1, 1)

        self.label_output_splatted = QLabel(self.groupBox_input_output)
        self.label_output_splatted.setObjectName(u"label_output_splatted")
        self.label_output_splatted.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_output_splatted, 2, 0, 1, 1)

        self.pushButton_select_source = QPushButton(self.groupBox_input_output)
        self.pushButton_select_source.setObjectName(u"pushButton_select_source")

        self.gridLayout_2.addWidget(self.pushButton_select_source, 0, 3, 1, 1)

        self.pushButton_browse_source = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_source.setObjectName(u"pushButton_browse_source")

        self.gridLayout_2.addWidget(self.pushButton_browse_source, 0, 2, 1, 1)

        self.lineEdit_input_depth = QLineEdit(self.groupBox_input_output)
        self.lineEdit_input_depth.setObjectName(u"lineEdit_input_depth")

        self.gridLayout_2.addWidget(self.lineEdit_input_depth, 1, 1, 1, 1)

        self.pushButton_browse_sidecar = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_sidecar.setObjectName(u"pushButton_browse_sidecar")

        self.gridLayout_2.addWidget(self.pushButton_browse_sidecar, 3, 2, 1, 1)

        self.lineEdit_inout_sidecar = QLineEdit(self.groupBox_input_output)
        self.lineEdit_inout_sidecar.setObjectName(u"lineEdit_inout_sidecar")

        self.gridLayout_2.addWidget(self.lineEdit_inout_sidecar, 3, 1, 1, 1)

        self.label_output_sidecar = QLabel(self.groupBox_input_output)
        self.label_output_sidecar.setObjectName(u"label_output_sidecar")
        self.label_output_sidecar.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_2.addWidget(self.label_output_sidecar, 3, 0, 1, 1)


        self.gridLayout_1.addWidget(self.groupBox_input_output, 0, 0, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_1, 0, 0, 1, 1)

        self.line_1 = QFrame(self.centralwidget)
        self.line_1.setObjectName(u"line_1")
        self.line_1.setFrameShape(QFrame.Shape.HLine)
        self.line_1.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line_1, 1, 0, 1, 1)

        self.horizontalLayout_frame_info = QHBoxLayout()
        self.horizontalLayout_frame_info.setObjectName(u"horizontalLayout_frame_info")
        self.label_frame_info = QLabel(self.centralwidget)
        self.label_frame_info.setObjectName(u"label_frame_info")

        self.horizontalLayout_frame_info.addWidget(self.label_frame_info)

        self.horizontalSlider = QSlider(self.centralwidget)
        self.horizontalSlider.setObjectName(u"horizontalSlider")
        self.horizontalSlider.setMinimumSize(QSize(0, 60))
        self.horizontalSlider.setStyleSheet(u"\n"
"	QSlider::groove:horizontal {\n"
"        height: 14px;\n"
"        background: palette(mid);\n"
"		border: 1px solid palette(dark);\n"
"        border-radius: 2px;\n"
"		margin: 0px; \n"
"    }\n"
"    \n"
"    QSlider::sub-page:horizontal {\n"
"        height: 14px;\n"
"        background: palette(mid);\n"
"		border: 1px solid palette(mid);\n"
"        border-radius: 2px;\n"
"		margin: 0px; \n"
"    }\n"
"    \n"
"    QSlider::handle:horizontal {\n"
"        background: #666666;\n"
"        width: 20px;\n"
"        height: 30px;\n"
"        margin: -10px 0;\n"
"        border-radius: 2px;\n"
"		border: 1px solid palette(mid);\n"
"    }\n"
"    \n"
"    QSlider::handle:horizontal:hover {\n"
"        background: palette(highlight);\n"
"        width: 20px;\n"
"        height: 30px;\n"
"        margin: -10px 0;\n"
"    }")
        self.horizontalSlider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_frame_info.addWidget(self.horizontalSlider)


        self.gridLayout.addLayout(self.horizontalLayout_frame_info, 2, 0, 1, 1)

        self.horizontalLayout_preview_controls = QHBoxLayout()
        self.horizontalLayout_preview_controls.setObjectName(u"horizontalLayout_preview_controls")
        self.label_preview_source = QLabel(self.centralwidget)
        self.label_preview_source.setObjectName(u"label_preview_source")

        self.horizontalLayout_preview_controls.addWidget(self.label_preview_source)

        self.comboBox_preview_source = QComboBox(self.centralwidget)
        self.comboBox_preview_source.addItem("")
        self.comboBox_preview_source.setObjectName(u"comboBox_preview_source")
        self.comboBox_preview_source.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_preview_controls.addWidget(self.comboBox_preview_source)

        self.pushButton_load_refresh = QPushButton(self.centralwidget)
        self.pushButton_load_refresh.setObjectName(u"pushButton_load_refresh")
        self.pushButton_load_refresh.setMinimumSize(QSize(140, 0))
        self.pushButton_load_refresh.setMaximumSize(QSize(140, 16777215))

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_load_refresh)

        self.horizontalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_preview_controls.addItem(self.horizontalSpacer_2)

        self.pushButton_prev = QPushButton(self.centralwidget)
        self.pushButton_prev.setObjectName(u"pushButton_prev")

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_prev)

        self.pushButton_next = QPushButton(self.centralwidget)
        self.pushButton_next.setObjectName(u"pushButton_next")

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_next)

        self.label_jump_to = QLabel(self.centralwidget)
        self.label_jump_to.setObjectName(u"label_jump_to")

        self.horizontalLayout_preview_controls.addWidget(self.label_jump_to)

        self.lineEdit_jump_to = QLineEdit(self.centralwidget)
        self.lineEdit_jump_to.setObjectName(u"lineEdit_jump_to")
        self.lineEdit_jump_to.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_preview_controls.addWidget(self.lineEdit_jump_to)

        self.label_video_info = QLabel(self.centralwidget)
        self.label_video_info.setObjectName(u"label_video_info")

        self.horizontalLayout_preview_controls.addWidget(self.label_video_info)

        self.pushButton_play = QPushButton(self.centralwidget)
        self.pushButton_play.setObjectName(u"pushButton_play")
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaybackStart))
        self.pushButton_play.setIcon(icon1)

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_play)

        self.pushButton_fast_forward = QPushButton(self.centralwidget)
        self.pushButton_fast_forward.setObjectName(u"pushButton_fast_forward")
        icon2 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaSeekForward))
        self.pushButton_fast_forward.setIcon(icon2)

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_fast_forward)

        self.spinBox_ff_speed = QSpinBox(self.centralwidget)
        self.spinBox_ff_speed.setObjectName(u"spinBox_ff_speed")
        self.spinBox_ff_speed.setMaximumSize(QSize(90, 16777215))
        self.spinBox_ff_speed.setValue(5)

        self.horizontalLayout_preview_controls.addWidget(self.spinBox_ff_speed)

        self.pushButton_loop_toggle = QPushButton(self.centralwidget)
        self.pushButton_loop_toggle.setObjectName(u"pushButton_loop_toggle")
        icon3 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaylistRepeat))
        self.pushButton_loop_toggle.setIcon(icon3)
        self.pushButton_loop_toggle.setCheckable(True)
        self.pushButton_loop_toggle.setChecked(False)
        self.pushButton_loop_toggle.setFlat(False)

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_loop_toggle)

        self.label_depth_map = QLabel(self.centralwidget)
        self.label_depth_map.setObjectName(u"label_depth_map")
        self.label_depth_map.setMinimumSize(QSize(70, 0))
        self.label_depth_map.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_preview_controls.addWidget(self.label_depth_map)

        self.comboBox_map_select = QComboBox(self.centralwidget)
        self.comboBox_map_select.addItem("")
        self.comboBox_map_select.addItem("")
        self.comboBox_map_select.setObjectName(u"comboBox_map_select")
        self.comboBox_map_select.setEnabled(False)

        self.horizontalLayout_preview_controls.addWidget(self.comboBox_map_select)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_preview_controls.addItem(self.horizontalSpacer_3)

        self.label_preview_scale = QLabel(self.centralwidget)
        self.label_preview_scale.setObjectName(u"label_preview_scale")

        self.horizontalLayout_preview_controls.addWidget(self.label_preview_scale)

        self.comboBox_preview_scale = QComboBox(self.centralwidget)
        self.comboBox_preview_scale.addItem("")
        self.comboBox_preview_scale.setObjectName(u"comboBox_preview_scale")
        self.comboBox_preview_scale.setMinimumSize(QSize(60, 0))

        self.horizontalLayout_preview_controls.addWidget(self.comboBox_preview_scale)


        self.gridLayout.addLayout(self.horizontalLayout_preview_controls, 3, 0, 1, 1)

        self.horizontalLayout_main_content = QHBoxLayout()
        self.horizontalLayout_main_content.setObjectName(u"horizontalLayout_main_content")
        self.verticalLayout_left_panel = QVBoxLayout()
        self.verticalLayout_left_panel.setObjectName(u"verticalLayout_left_panel")
        self.groupBox_process_resolution = QGroupBox(self.centralwidget)
        self.groupBox_process_resolution.setObjectName(u"groupBox_process_resolution")
        self.groupBox_process_resolution.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_3 = QGridLayout(self.groupBox_process_resolution)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.checkBox_enable_full_res = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_enable_full_res.setObjectName(u"checkBox_enable_full_res")
        self.checkBox_enable_full_res.setChecked(True)

        self.gridLayout_3.addWidget(self.checkBox_enable_full_res, 0, 0, 1, 2)

        self.label_batch_size_full = QLabel(self.groupBox_process_resolution)
        self.label_batch_size_full.setObjectName(u"label_batch_size_full")

        self.gridLayout_3.addWidget(self.label_batch_size_full, 0, 2, 1, 1)

        self.checkBox_enable_low_res = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_enable_low_res.setObjectName(u"checkBox_enable_low_res")
        self.checkBox_enable_low_res.setChecked(True)

        self.gridLayout_3.addWidget(self.checkBox_enable_low_res, 1, 0, 1, 2)

        self.label_batch_size_low = QLabel(self.groupBox_process_resolution)
        self.label_batch_size_low.setObjectName(u"label_batch_size_low")

        self.gridLayout_3.addWidget(self.label_batch_size_low, 1, 2, 1, 1)

        self.label_width = QLabel(self.groupBox_process_resolution)
        self.label_width.setObjectName(u"label_width")

        self.gridLayout_3.addWidget(self.label_width, 2, 0, 1, 1)

        self.label_height = QLabel(self.groupBox_process_resolution)
        self.label_height.setObjectName(u"label_height")

        self.gridLayout_3.addWidget(self.label_height, 2, 2, 1, 1)

        self.checkBox_dual_output = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_dual_output.setObjectName(u"checkBox_dual_output")

        self.gridLayout_3.addWidget(self.checkBox_dual_output, 0, 4, 1, 1)

        self.checkBox_ffmpeg = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_ffmpeg.setObjectName(u"checkBox_ffmpeg")

        self.gridLayout_3.addWidget(self.checkBox_ffmpeg, 1, 4, 1, 1)

        self.lineEdit_high_batch = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_high_batch.setObjectName(u"lineEdit_high_batch")

        self.gridLayout_3.addWidget(self.lineEdit_high_batch, 0, 3, 1, 1)

        self.lineEdit_low_batch = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_low_batch.setObjectName(u"lineEdit_low_batch")

        self.gridLayout_3.addWidget(self.lineEdit_low_batch, 1, 3, 1, 1)

        self.lineEdit_low_height = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_low_height.setObjectName(u"lineEdit_low_height")

        self.gridLayout_3.addWidget(self.lineEdit_low_height, 2, 3, 1, 1)

        self.lineEdit_low_width = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_low_width.setObjectName(u"lineEdit_low_width")

        self.gridLayout_3.addWidget(self.lineEdit_low_width, 2, 1, 1, 1)


        self.verticalLayout_left_panel.addWidget(self.groupBox_process_resolution)

        self.groupBox_splatting_settings = QGroupBox(self.centralwidget)
        self.groupBox_splatting_settings.setObjectName(u"groupBox_splatting_settings")
        self.groupBox_splatting_settings.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_4 = QGridLayout(self.groupBox_splatting_settings)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.label_border = QLabel(self.groupBox_splatting_settings)
        self.label_border.setObjectName(u"label_border")

        self.gridLayout_4.addWidget(self.label_border, 1, 2, 1, 1)

        self.lineEdit_mesh_extrusion = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_extrusion.setObjectName(u"lineEdit_mesh_extrusion")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_extrusion, 3, 1, 1, 1)

        self.label_mask = QLabel(self.groupBox_splatting_settings)
        self.label_mask.setObjectName(u"label_mask")

        self.gridLayout_4.addWidget(self.label_mask, 1, 0, 1, 1)

        self.comboBox_auto_convergence = QComboBox(self.groupBox_splatting_settings)
        self.comboBox_auto_convergence.addItem("")
        self.comboBox_auto_convergence.setObjectName(u"comboBox_auto_convergence")

        self.gridLayout_4.addWidget(self.comboBox_auto_convergence, 0, 3, 1, 1)

        self.lineEdit_process_length = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_process_length.setObjectName(u"lineEdit_process_length")

        self.gridLayout_4.addWidget(self.lineEdit_process_length, 0, 1, 1, 1)

        self.label_density = QLabel(self.groupBox_splatting_settings)
        self.label_density.setObjectName(u"label_density")

        self.gridLayout_4.addWidget(self.label_density, 3, 2, 1, 1)

        self.lineEdit_mesh_dolly = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_dolly.setObjectName(u"lineEdit_mesh_dolly")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_dolly, 2, 3, 1, 1)

        self.label_extrusion = QLabel(self.groupBox_splatting_settings)
        self.label_extrusion.setObjectName(u"label_extrusion")

        self.gridLayout_4.addWidget(self.label_extrusion, 3, 0, 1, 1)

        self.label_auto_convergence = QLabel(self.groupBox_splatting_settings)
        self.label_auto_convergence.setObjectName(u"label_auto_convergence")

        self.gridLayout_4.addWidget(self.label_auto_convergence, 0, 2, 1, 1)

        self.label_process_length = QLabel(self.groupBox_splatting_settings)
        self.label_process_length.setObjectName(u"label_process_length")

        self.gridLayout_4.addWidget(self.label_process_length, 0, 0, 1, 1)

        self.comboBox_mask_type = QComboBox(self.groupBox_splatting_settings)
        self.comboBox_mask_type.addItem("")
        self.comboBox_mask_type.setObjectName(u"comboBox_mask_type")

        self.gridLayout_4.addWidget(self.comboBox_mask_type, 1, 1, 1, 1)

        self.comboBox_border = QComboBox(self.groupBox_splatting_settings)
        self.comboBox_border.addItem("")
        self.comboBox_border.setObjectName(u"comboBox_border")

        self.gridLayout_4.addWidget(self.comboBox_border, 1, 3, 1, 1)

        self.label_mesh_bias = QLabel(self.groupBox_splatting_settings)
        self.label_mesh_bias.setObjectName(u"label_mesh_bias")

        self.gridLayout_4.addWidget(self.label_mesh_bias, 2, 0, 1, 1)

        self.lineEdit_mesh_bias = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_bias.setObjectName(u"lineEdit_mesh_bias")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_bias, 2, 1, 1, 1)

        self.label_dolly = QLabel(self.groupBox_splatting_settings)
        self.label_dolly.setObjectName(u"label_dolly")

        self.gridLayout_4.addWidget(self.label_dolly, 2, 2, 1, 1)

        self.lineEdit_mesh_density = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_density.setObjectName(u"lineEdit_mesh_density")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_density, 3, 3, 1, 1)


        self.verticalLayout_left_panel.addWidget(self.groupBox_splatting_settings)


        self.horizontalLayout_main_content.addLayout(self.verticalLayout_left_panel)

        self.verticalLayout_right_panel = QVBoxLayout()
        self.verticalLayout_right_panel.setObjectName(u"verticalLayout_right_panel")
        self.groupBox_depth_map_preprocessing = QGroupBox(self.centralwidget)
        self.groupBox_depth_map_preprocessing.setObjectName(u"groupBox_depth_map_preprocessing")
        self.gridLayout_5 = QGridLayout(self.groupBox_depth_map_preprocessing)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(-1, 6, -1, -1)
        self.label_blur_left_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_left_value.setObjectName(u"label_blur_left_value")
        self.label_blur_left_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_blur_left_value, 2, 5, 1, 1)

        self.horizontalSlider_blur_y = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_blur_y.setObjectName(u"horizontalSlider_blur_y")
        self.horizontalSlider_blur_y.setValue(5)
        self.horizontalSlider_blur_y.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_blur_y, 1, 4, 1, 1)

        self.label_blur_y_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_y_value.setObjectName(u"label_blur_y_value")
        self.label_blur_y_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_blur_y_value, 1, 5, 1, 1)

        self.horizontalSlider_blur_left = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_blur_left.setObjectName(u"horizontalSlider_blur_left")
        self.horizontalSlider_blur_left.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_blur_left, 2, 4, 1, 1)

        self.label_blur_y = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_y.setObjectName(u"label_blur_y")
        self.label_blur_y.setMinimumSize(QSize(44, 0))
        self.label_blur_y.setMaximumSize(QSize(40, 16777215))
        self.label_blur_y.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_blur_y, 1, 3, 1, 1)

        self.label_dilate_left = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_left.setObjectName(u"label_dilate_left")
        self.label_dilate_left.setMinimumSize(QSize(44, 0))
        self.label_dilate_left.setMaximumSize(QSize(40, 16777215))
        self.label_dilate_left.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_dilate_left, 2, 0, 1, 1)

        self.horizontalSlider_dilate_left = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_dilate_left.setObjectName(u"horizontalSlider_dilate_left")
        self.horizontalSlider_dilate_left.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_dilate_left, 2, 1, 1, 1)

        self.label_dilate_y = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_y.setObjectName(u"label_dilate_y")
        self.label_dilate_y.setMinimumSize(QSize(44, 0))
        self.label_dilate_y.setMaximumSize(QSize(40, 16777215))
        self.label_dilate_y.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_dilate_y, 0, 3, 1, 1)

        self.label_blur_left = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_left.setObjectName(u"label_blur_left")
        self.label_blur_left.setMinimumSize(QSize(44, 0))
        self.label_blur_left.setMaximumSize(QSize(40, 16777215))
        self.label_blur_left.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_blur_left, 2, 3, 1, 1)

        self.label_dilate_x = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_x.setObjectName(u"label_dilate_x")
        self.label_dilate_x.setMinimumSize(QSize(44, 0))
        self.label_dilate_x.setMaximumSize(QSize(40, 16777215))
        self.label_dilate_x.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_dilate_x, 0, 0, 1, 1)

        self.label_blur_x = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_x.setObjectName(u"label_blur_x")
        self.label_blur_x.setMinimumSize(QSize(44, 0))
        self.label_blur_x.setMaximumSize(QSize(40, 16777215))
        self.label_blur_x.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_5.addWidget(self.label_blur_x, 1, 0, 1, 1)

        self.label_blue_x_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blue_x_value.setObjectName(u"label_blue_x_value")
        self.label_blue_x_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_blue_x_value, 1, 2, 1, 1)

        self.label_dilate_y_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_y_value.setObjectName(u"label_dilate_y_value")
        self.label_dilate_y_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_dilate_y_value, 0, 5, 1, 1)

        self.lineEdit_blur_bias = QLineEdit(self.groupBox_depth_map_preprocessing)
        self.lineEdit_blur_bias.setObjectName(u"lineEdit_blur_bias")
        self.lineEdit_blur_bias.setMaximumSize(QSize(30, 16777215))

        self.gridLayout_5.addWidget(self.lineEdit_blur_bias, 2, 6, 1, 1)

        self.horizontalSlider_blur_x = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_blur_x.setObjectName(u"horizontalSlider_blur_x")
        self.horizontalSlider_blur_x.setValue(5)
        self.horizontalSlider_blur_x.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_blur_x, 1, 1, 1, 1)

        self.horizontalSlider_dilate_x = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_dilate_x.setObjectName(u"horizontalSlider_dilate_x")
        self.horizontalSlider_dilate_x.setValue(12)
        self.horizontalSlider_dilate_x.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_dilate_x, 0, 1, 1, 1)

        self.horizontalSlider_dilate_y = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_dilate_y.setObjectName(u"horizontalSlider_dilate_y")
        self.horizontalSlider_dilate_y.setValue(3)
        self.horizontalSlider_dilate_y.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_dilate_y, 0, 4, 1, 1)

        self.label_dilate_left_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_left_value.setObjectName(u"label_dilate_left_value")
        self.label_dilate_left_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_dilate_left_value, 2, 2, 1, 1)

        self.label_dilate_x_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_x_value.setObjectName(u"label_dilate_x_value")
        self.label_dilate_x_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_dilate_x_value, 0, 2, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)

        self.verticalLayout_right_panel.addWidget(self.groupBox_depth_map_preprocessing)

        self.groupBox_stereo_projection = QGroupBox(self.centralwidget)
        self.groupBox_stereo_projection.setObjectName(u"groupBox_stereo_projection")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_stereo_projection.sizePolicy().hasHeightForWidth())
        self.groupBox_stereo_projection.setSizePolicy(sizePolicy)
        self.gridLayout_8 = QGridLayout(self.groupBox_stereo_projection)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.label_gamma = QLabel(self.groupBox_stereo_projection)
        self.label_gamma.setObjectName(u"label_gamma")
        self.label_gamma.setMinimumSize(QSize(80, 0))
        self.label_gamma.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_8.addWidget(self.label_gamma, 0, 0, 1, 1)

        self.horizontalSlider_gamma = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_gamma.setObjectName(u"horizontalSlider_gamma")
        self.horizontalSlider_gamma.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_gamma.setValue(99)
        self.horizontalSlider_gamma.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_gamma, 0, 1, 1, 1)

        self.label_gamma_value = QLabel(self.groupBox_stereo_projection)
        self.label_gamma_value.setObjectName(u"label_gamma_value")
        self.label_gamma_value.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.label_gamma_value, 0, 2, 1, 1)

        self.label_disparity = QLabel(self.groupBox_stereo_projection)
        self.label_disparity.setObjectName(u"label_disparity")
        self.label_disparity.setMinimumSize(QSize(70, 0))
        self.label_disparity.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_8.addWidget(self.label_disparity, 0, 3, 1, 1)

        self.horizontalSlider_disparity = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_disparity.setObjectName(u"horizontalSlider_disparity")
        self.horizontalSlider_disparity.setMinimumSize(QSize(96, 0))
        self.horizontalSlider_disparity.setValue(45)
        self.horizontalSlider_disparity.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_disparity, 0, 4, 1, 1)

        self.label_disparity_value = QLabel(self.groupBox_stereo_projection)
        self.label_disparity_value.setObjectName(u"label_disparity_value")
        self.label_disparity_value.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.label_disparity_value, 0, 5, 1, 1)

        self.label_convergence = QLabel(self.groupBox_stereo_projection)
        self.label_convergence.setObjectName(u"label_convergence")
        self.label_convergence.setMinimumSize(QSize(80, 0))
        self.label_convergence.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_8.addWidget(self.label_convergence, 1, 0, 1, 1)

        self.horizontalSlider_convergence = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_convergence.setObjectName(u"horizontalSlider_convergence")
        self.horizontalSlider_convergence.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_convergence.setValue(41)
        self.horizontalSlider_convergence.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_convergence, 1, 1, 1, 4)

        self.label_convergence_value = QLabel(self.groupBox_stereo_projection)
        self.label_convergence_value.setObjectName(u"label_convergence_value")
        self.label_convergence_value.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.label_convergence_value, 1, 5, 1, 1)

        self.label_border_width = QLabel(self.groupBox_stereo_projection)
        self.label_border_width.setObjectName(u"label_border_width")
        self.label_border_width.setMinimumSize(QSize(80, 0))
        self.label_border_width.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_8.addWidget(self.label_border_width, 2, 0, 1, 1)

        self.horizontalSlider_border_width = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_border_width.setObjectName(u"horizontalSlider_border_width")
        self.horizontalSlider_border_width.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_border_width.setValue(99)
        self.horizontalSlider_border_width.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_border_width, 2, 1, 1, 1)

        self.label_border_width_value = QLabel(self.groupBox_stereo_projection)
        self.label_border_width_value.setObjectName(u"label_border_width_value")
        self.label_border_width_value.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.label_border_width_value, 2, 2, 1, 1)

        self.label_bias = QLabel(self.groupBox_stereo_projection)
        self.label_bias.setObjectName(u"label_bias")
        self.label_bias.setMinimumSize(QSize(70, 0))
        self.label_bias.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout_8.addWidget(self.label_bias, 2, 3, 1, 1)

        self.horizontalSlider_border_bias = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_border_bias.setObjectName(u"horizontalSlider_border_bias")
        self.horizontalSlider_border_bias.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_border_bias.setValue(27)
        self.horizontalSlider_border_bias.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_border_bias, 2, 4, 1, 1)

        self.label_bias_value = QLabel(self.groupBox_stereo_projection)
        self.label_bias_value.setObjectName(u"label_bias_value")
        self.label_bias_value.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.label_bias_value, 2, 5, 1, 1)

        self.checkBox_normalization = QCheckBox(self.groupBox_stereo_projection)
        self.checkBox_normalization.setObjectName(u"checkBox_normalization")
        self.checkBox_normalization.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.checkBox_normalization, 3, 1, 1, 1)

        self.checkBox_resume = QCheckBox(self.groupBox_stereo_projection)
        self.checkBox_resume.setObjectName(u"checkBox_resume")
        self.checkBox_resume.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.checkBox_resume, 3, 3, 1, 1)

        self.checkBox_cross_view = QCheckBox(self.groupBox_stereo_projection)
        self.checkBox_cross_view.setObjectName(u"checkBox_cross_view")

        self.gridLayout_8.addWidget(self.checkBox_cross_view, 3, 4, 1, 1)

        self.gridLayout_8.setColumnStretch(0, 1)
        self.gridLayout_8.setColumnStretch(1, 4)
        self.gridLayout_8.setColumnStretch(2, 1)
        self.gridLayout_8.setColumnStretch(3, 1)
        self.gridLayout_8.setColumnStretch(4, 4)
        self.gridLayout_8.setColumnStretch(5, 1)

        self.verticalLayout_right_panel.addWidget(self.groupBox_stereo_projection)


        self.horizontalLayout_main_content.addLayout(self.verticalLayout_right_panel)

        self.verticalLayout_info_panel = QVBoxLayout()
        self.verticalLayout_info_panel.setObjectName(u"verticalLayout_info_panel")
        self.groupBox_current_processing_info = QGroupBox(self.centralwidget)
        self.groupBox_current_processing_info.setObjectName(u"groupBox_current_processing_info")
        self.gridLayout_7 = QGridLayout(self.groupBox_current_processing_info)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.label_info_gamma_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_gamma_value.setObjectName(u"label_info_gamma_value")

        self.gridLayout_7.addWidget(self.label_info_gamma_value, 4, 1, 1, 1)

        self.label_info_map_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_map_value.setObjectName(u"label_info_map_value")

        self.gridLayout_7.addWidget(self.label_info_map_value, 2, 3, 1, 1)

        self.label_info_frames_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_frames_value.setObjectName(u"label_info_frames_value")

        self.gridLayout_7.addWidget(self.label_info_frames_value, 3, 1, 1, 1)

        self.label_info_disparity_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_disparity_value.setObjectName(u"label_info_disparity_value")

        self.gridLayout_7.addWidget(self.label_info_disparity_value, 0, 3, 1, 1)

        self.label_gamma_info = QLabel(self.groupBox_current_processing_info)
        self.label_gamma_info.setObjectName(u"label_gamma_info")

        self.gridLayout_7.addWidget(self.label_gamma_info, 4, 0, 1, 1)

        self.label_convergence_info = QLabel(self.groupBox_current_processing_info)
        self.label_convergence_info.setObjectName(u"label_convergence_info")

        self.gridLayout_7.addWidget(self.label_convergence_info, 1, 2, 1, 1)

        self.label_filename_info = QLabel(self.groupBox_current_processing_info)
        self.label_filename_info.setObjectName(u"label_filename_info")

        self.gridLayout_7.addWidget(self.label_filename_info, 0, 0, 1, 1)

        self.label_resolution_info = QLabel(self.groupBox_current_processing_info)
        self.label_resolution_info.setObjectName(u"label_resolution_info")

        self.gridLayout_7.addWidget(self.label_resolution_info, 2, 0, 1, 1)

        self.label_task_info = QLabel(self.groupBox_current_processing_info)
        self.label_task_info.setObjectName(u"label_task_info")

        self.gridLayout_7.addWidget(self.label_task_info, 1, 0, 1, 1)

        self.label_frames_info = QLabel(self.groupBox_current_processing_info)
        self.label_frames_info.setObjectName(u"label_frames_info")

        self.gridLayout_7.addWidget(self.label_frames_info, 3, 0, 1, 1)

        self.label_info_filename_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_filename_value.setObjectName(u"label_info_filename_value")

        self.gridLayout_7.addWidget(self.label_info_filename_value, 0, 1, 1, 1)

        self.label_info_convergence_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_convergence_value.setObjectName(u"label_info_convergence_value")

        self.gridLayout_7.addWidget(self.label_info_convergence_value, 1, 3, 1, 1)

        self.label_info_resolution_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_resolution_value.setObjectName(u"label_info_resolution_value")

        self.gridLayout_7.addWidget(self.label_info_resolution_value, 2, 1, 1, 1)

        self.label_disparity_info = QLabel(self.groupBox_current_processing_info)
        self.label_disparity_info.setObjectName(u"label_disparity_info")

        self.gridLayout_7.addWidget(self.label_disparity_info, 0, 2, 1, 1)

        self.label_map_info = QLabel(self.groupBox_current_processing_info)
        self.label_map_info.setObjectName(u"label_map_info")

        self.gridLayout_7.addWidget(self.label_map_info, 2, 2, 1, 1)

        self.label_info_task_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_task_value.setObjectName(u"label_info_task_value")

        self.gridLayout_7.addWidget(self.label_info_task_value, 1, 1, 1, 1)

        self.gridLayout_7.setColumnStretch(0, 2)
        self.gridLayout_7.setColumnStretch(1, 4)
        self.gridLayout_7.setColumnStretch(2, 2)
        self.gridLayout_7.setColumnStretch(3, 4)

        self.verticalLayout_info_panel.addWidget(self.groupBox_current_processing_info)

        self.groupBox_dev_tools = QGroupBox(self.centralwidget)
        self.groupBox_dev_tools.setObjectName(u"groupBox_dev_tools")
        self.gridLayout_6 = QGridLayout(self.groupBox_dev_tools)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.checkBox_crosshair = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_crosshair.setObjectName(u"checkBox_crosshair")
        self.checkBox_crosshair.setEnabled(True)
        self.checkBox_crosshair.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_crosshair, 1, 0, 1, 1)

        self.checkBox_dp = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_dp.setObjectName(u"checkBox_dp")
        self.checkBox_dp.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_dp, 1, 3, 1, 1)

        self.checkBox_multi = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_multi.setObjectName(u"checkBox_multi")
        self.checkBox_multi.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_multi, 1, 2, 1, 1)

        self.checkBox_splat_test = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_splat_test.setObjectName(u"checkBox_splat_test")

        self.gridLayout_6.addWidget(self.checkBox_splat_test, 0, 2, 1, 1)

        self.checkBox_white = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_white.setObjectName(u"checkBox_white")
        self.checkBox_white.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_white, 1, 1, 1, 1)

        self.checkBox_true_max = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_true_max.setObjectName(u"checkBox_true_max")

        self.gridLayout_6.addWidget(self.checkBox_true_max, 0, 3, 1, 1)

        self.checkBox_map_test = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_map_test.setObjectName(u"checkBox_map_test")

        self.gridLayout_6.addWidget(self.checkBox_map_test, 0, 1, 1, 1)

        self.checkBox_skip_low_res = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_skip_low_res.setObjectName(u"checkBox_skip_low_res")

        self.gridLayout_6.addWidget(self.checkBox_skip_low_res, 0, 0, 1, 1)


        self.verticalLayout_info_panel.addWidget(self.groupBox_dev_tools)


        self.horizontalLayout_main_content.addLayout(self.verticalLayout_info_panel)

        self.horizontalLayout_main_content.setStretch(0, 1)
        self.horizontalLayout_main_content.setStretch(1, 5)
        self.horizontalLayout_main_content.setStretch(2, 2)

        self.gridLayout.addLayout(self.horizontalLayout_main_content, 4, 0, 1, 1)

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setValue(0)

        self.gridLayout.addWidget(self.progressBar, 5, 0, 1, 1)

        self.label_status = QLabel(self.centralwidget)
        self.label_status.setObjectName(u"label_status")
        self.label_status.setMinimumSize(QSize(0, 50))
        self.label_status.setMaximumSize(QSize(16777215, 50))
        self.label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label_status, 6, 0, 1, 1)

        self.horizontalLayout_bottom_buttons = QHBoxLayout()
        self.horizontalLayout_bottom_buttons.setObjectName(u"horizontalLayout_bottom_buttons")
        self.pushButton_single = QPushButton(self.centralwidget)
        self.pushButton_single.setObjectName(u"pushButton_single")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_single)

        self.pushButton_start = QPushButton(self.centralwidget)
        self.pushButton_start.setObjectName(u"pushButton_start")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_start)

        self.label_from = QLabel(self.centralwidget)
        self.label_from.setObjectName(u"label_from")

        self.horizontalLayout_bottom_buttons.addWidget(self.label_from)

        self.spinBox_from = QSpinBox(self.centralwidget)
        self.spinBox_from.setObjectName(u"spinBox_from")

        self.horizontalLayout_bottom_buttons.addWidget(self.spinBox_from)

        self.label_to = QLabel(self.centralwidget)
        self.label_to.setObjectName(u"label_to")

        self.horizontalLayout_bottom_buttons.addWidget(self.label_to)

        self.spinBox_to = QSpinBox(self.centralwidget)
        self.spinBox_to.setObjectName(u"spinBox_to")

        self.horizontalLayout_bottom_buttons.addWidget(self.spinBox_to)

        self.pushButton_stop = QPushButton(self.centralwidget)
        self.pushButton_stop.setObjectName(u"pushButton_stop")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_stop)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_bottom_buttons.addItem(self.horizontalSpacer_4)

        self.pushButton_preview_auto_converge = QPushButton(self.centralwidget)
        self.pushButton_preview_auto_converge.setObjectName(u"pushButton_preview_auto_converge")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_preview_auto_converge)

        self.pushButton_auto_pass = QPushButton(self.centralwidget)
        self.pushButton_auto_pass.setObjectName(u"pushButton_auto_pass")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_auto_pass)

        self.pushButton_update_sidecar = QPushButton(self.centralwidget)
        self.pushButton_update_sidecar.setObjectName(u"pushButton_update_sidecar")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_update_sidecar)


        self.gridLayout.addLayout(self.horizontalLayout_bottom_buttons, 7, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1309, 33))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuOptions = QMenu(self.menubar)
        self.menuOptions.setObjectName(u"menuOptions")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QWidget.setTabOrder(self.lineEdit_input_source, self.pushButton_browse_source)
        QWidget.setTabOrder(self.pushButton_browse_source, self.pushButton_select_source)
        QWidget.setTabOrder(self.pushButton_select_source, self.lineEdit_input_depth)
        QWidget.setTabOrder(self.lineEdit_input_depth, self.pushButton_browse_depth)
        QWidget.setTabOrder(self.pushButton_browse_depth, self.pushButton_select_depth)
        QWidget.setTabOrder(self.pushButton_select_depth, self.lineEdit_output_splatted)
        QWidget.setTabOrder(self.lineEdit_output_splatted, self.pushButton_browse_output)
        QWidget.setTabOrder(self.pushButton_browse_output, self.checkBox_multi_map)
        QWidget.setTabOrder(self.checkBox_multi_map, self.lineEdit_inout_sidecar)
        QWidget.setTabOrder(self.lineEdit_inout_sidecar, self.pushButton_browse_sidecar)
        QWidget.setTabOrder(self.pushButton_browse_sidecar, self.horizontalSlider)
        QWidget.setTabOrder(self.horizontalSlider, self.comboBox_preview_source)
        QWidget.setTabOrder(self.comboBox_preview_source, self.pushButton_load_refresh)
        QWidget.setTabOrder(self.pushButton_load_refresh, self.pushButton_prev)
        QWidget.setTabOrder(self.pushButton_prev, self.pushButton_next)
        QWidget.setTabOrder(self.pushButton_next, self.lineEdit_jump_to)
        QWidget.setTabOrder(self.lineEdit_jump_to, self.pushButton_play)
        QWidget.setTabOrder(self.pushButton_play, self.pushButton_fast_forward)
        QWidget.setTabOrder(self.pushButton_fast_forward, self.spinBox_ff_speed)
        QWidget.setTabOrder(self.spinBox_ff_speed, self.pushButton_loop_toggle)
        QWidget.setTabOrder(self.pushButton_loop_toggle, self.comboBox_map_select)
        QWidget.setTabOrder(self.comboBox_map_select, self.checkBox_enable_full_res)
        QWidget.setTabOrder(self.checkBox_enable_full_res, self.lineEdit_high_batch)
        QWidget.setTabOrder(self.lineEdit_high_batch, self.checkBox_dual_output)
        QWidget.setTabOrder(self.checkBox_dual_output, self.checkBox_enable_low_res)
        QWidget.setTabOrder(self.checkBox_enable_low_res, self.lineEdit_low_batch)
        QWidget.setTabOrder(self.lineEdit_low_batch, self.checkBox_ffmpeg)
        QWidget.setTabOrder(self.checkBox_ffmpeg, self.lineEdit_low_width)
        QWidget.setTabOrder(self.lineEdit_low_width, self.lineEdit_low_height)
        QWidget.setTabOrder(self.lineEdit_low_height, self.horizontalSlider_dilate_x)
        QWidget.setTabOrder(self.horizontalSlider_dilate_x, self.horizontalSlider_dilate_y)
        QWidget.setTabOrder(self.horizontalSlider_dilate_y, self.horizontalSlider_blur_x)
        QWidget.setTabOrder(self.horizontalSlider_blur_x, self.horizontalSlider_blur_y)
        QWidget.setTabOrder(self.horizontalSlider_blur_y, self.horizontalSlider_dilate_left)
        QWidget.setTabOrder(self.horizontalSlider_dilate_left, self.horizontalSlider_blur_left)
        QWidget.setTabOrder(self.horizontalSlider_blur_left, self.lineEdit_process_length)
        QWidget.setTabOrder(self.lineEdit_process_length, self.comboBox_auto_convergence)
        QWidget.setTabOrder(self.comboBox_auto_convergence, self.comboBox_mask_type)
        QWidget.setTabOrder(self.comboBox_mask_type, self.comboBox_border)
        QWidget.setTabOrder(self.comboBox_border, self.lineEdit_mesh_bias)
        QWidget.setTabOrder(self.lineEdit_mesh_bias, self.lineEdit_mesh_dolly)
        QWidget.setTabOrder(self.lineEdit_mesh_dolly, self.lineEdit_mesh_extrusion)
        QWidget.setTabOrder(self.lineEdit_mesh_extrusion, self.lineEdit_mesh_density)
        QWidget.setTabOrder(self.lineEdit_mesh_density, self.checkBox_skip_low_res)
        QWidget.setTabOrder(self.checkBox_skip_low_res, self.checkBox_map_test)
        QWidget.setTabOrder(self.checkBox_map_test, self.checkBox_splat_test)
        QWidget.setTabOrder(self.checkBox_splat_test, self.checkBox_true_max)
        QWidget.setTabOrder(self.checkBox_true_max, self.pushButton_single)
        QWidget.setTabOrder(self.pushButton_single, self.pushButton_start)
        QWidget.setTabOrder(self.pushButton_start, self.spinBox_from)
        QWidget.setTabOrder(self.spinBox_from, self.spinBox_to)
        QWidget.setTabOrder(self.spinBox_to, self.pushButton_stop)
        QWidget.setTabOrder(self.pushButton_stop, self.pushButton_preview_auto_converge)
        QWidget.setTabOrder(self.pushButton_preview_auto_converge, self.pushButton_auto_pass)
        QWidget.setTabOrder(self.pushButton_auto_pass, self.pushButton_update_sidecar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.action_load_settings)
        self.menuFile.addAction(self.action_save)
        self.menuFile.addAction(self.action_save_to_file)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_load_fsexport)
        self.menuFile.addAction(self.action_fsexport_to_sidecar)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_update_from_sidecar)
        self.menuFile.addAction(self.action_auto_update_sidecar)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_restore_from_finished)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_exit)
        self.menuOptions.addAction(self.action_encoder)
        self.menuOptions.addSeparator()
        self.menuHelp.addAction(self.action_guide)
        self.menuHelp.addAction(self.action_calculator)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.action_debug)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.action_about)

        self.retranslateUi(MainWindow)

        self.pushButton_loop_toggle.setDefault(False)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Stereocrafter Splatting (Batch) 26-03-29.0", None))
        self.action_load_settings.setText(QCoreApplication.translate("MainWindow", u"Load settings from File...", None))
        self.action_save.setText(QCoreApplication.translate("MainWindow", u"Save Settings", None))
#if QT_CONFIG(shortcut)
        self.action_save.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.action_save_to_file.setText(QCoreApplication.translate("MainWindow", u"Save Settings to File...", None))
        self.action_load_fsexport.setText(QCoreApplication.translate("MainWindow", u"Load Fusion Export (,fsexport)...", None))
        self.action_fsexport_to_sidecar.setText(QCoreApplication.translate("MainWindow", u"FSExport to Custom Sidecar...", None))
        self.action_restore_from_finished.setText(QCoreApplication.translate("MainWindow", u"Restore Finished Files", None))
        self.action_encoder.setText(QCoreApplication.translate("MainWindow", u"Encoder Settings", None))
        self.action_update_from_sidecar.setText(QCoreApplication.translate("MainWindow", u"Load Sidecar Data", None))
        self.action_auto_update_sidecar.setText(QCoreApplication.translate("MainWindow", u"Save Sidecar on Next", None))
        self.action_guide.setText(QCoreApplication.translate("MainWindow", u"User Guide", None))
        self.action_calculator.setText(QCoreApplication.translate("MainWindow", u"VRAM Calculator", None))
        self.action_debug.setText(QCoreApplication.translate("MainWindow", u"Debug Logging", None))
        self.action_about.setText(QCoreApplication.translate("MainWindow", u"About Splatting GUI", None))
        self.action_exit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.groupBox_input_output.setTitle(QCoreApplication.translate("MainWindow", u"Input/Output Folders", None))
        self.pushButton_browse_output.setText(QCoreApplication.translate("MainWindow", u"Browse Folder", None))
        self.label_input_depth.setText(QCoreApplication.translate("MainWindow", u"Input Depth Maps:", None))
        self.pushButton_browse_depth.setText(QCoreApplication.translate("MainWindow", u"Browse Folder", None))
        self.lineEdit_input_source.setText(QCoreApplication.translate("MainWindow", u"./workspace/clips", None))
        self.pushButton_select_depth.setText(QCoreApplication.translate("MainWindow", u"Select File", None))
        self.checkBox_multi_map.setText(QCoreApplication.translate("MainWindow", u"Multi-Map", None))
        self.lineEdit_output_splatted.setText(QCoreApplication.translate("MainWindow", u"./workspace/splat", None))
        self.label_input_source.setText(QCoreApplication.translate("MainWindow", u"Input Source Clips:", None))
        self.label_output_splatted.setText(QCoreApplication.translate("MainWindow", u"Output Splatted:", None))
        self.pushButton_select_source.setText(QCoreApplication.translate("MainWindow", u"Select File", None))
        self.pushButton_browse_source.setText(QCoreApplication.translate("MainWindow", u"Browse Folder", None))
        self.lineEdit_input_depth.setText(QCoreApplication.translate("MainWindow", u"./workspace/depth", None))
        self.pushButton_browse_sidecar.setText(QCoreApplication.translate("MainWindow", u"Browse Folder", None))
        self.lineEdit_inout_sidecar.setText(QCoreApplication.translate("MainWindow", u"./workspace/sidecar", None))
        self.label_output_sidecar.setText(QCoreApplication.translate("MainWindow", u"I/O Sidecar:", None))
        self.label_frame_info.setText(QCoreApplication.translate("MainWindow", u"Frame: 0 / 0", None))
        self.label_preview_source.setText(QCoreApplication.translate("MainWindow", u"Preview Source:", None))
        self.comboBox_preview_source.setItemText(0, QCoreApplication.translate("MainWindow", u"Splat Result", None))

        self.pushButton_load_refresh.setText(QCoreApplication.translate("MainWindow", u"Load/Refresh List", None))
#if QT_CONFIG(shortcut)
        self.pushButton_load_refresh.setShortcut(QCoreApplication.translate("MainWindow", u"Return", None))
#endif // QT_CONFIG(shortcut)
        self.pushButton_prev.setText(QCoreApplication.translate("MainWindow", u"< Prev", None))
#if QT_CONFIG(shortcut)
        self.pushButton_prev.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Left", None))
#endif // QT_CONFIG(shortcut)
        self.pushButton_next.setText(QCoreApplication.translate("MainWindow", u"Next >", None))
#if QT_CONFIG(shortcut)
        self.pushButton_next.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Right", None))
#endif // QT_CONFIG(shortcut)
        self.label_jump_to.setText(QCoreApplication.translate("MainWindow", u"Jump to:", None))
        self.lineEdit_jump_to.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_video_info.setText(QCoreApplication.translate("MainWindow", u"Video: 0 / 0", None))
        self.pushButton_play.setText("")
        self.pushButton_fast_forward.setText("")
        self.pushButton_loop_toggle.setText("")
        self.label_depth_map.setText(QCoreApplication.translate("MainWindow", u"Depth Map:", None))
        self.comboBox_map_select.setItemText(0, QCoreApplication.translate("MainWindow", u"Default", None))
        self.comboBox_map_select.setItemText(1, "")

        self.label_preview_scale.setText(QCoreApplication.translate("MainWindow", u"Preview Scale:", None))
        self.comboBox_preview_scale.setItemText(0, QCoreApplication.translate("MainWindow", u"70%", None))

        self.groupBox_process_resolution.setTitle(QCoreApplication.translate("MainWindow", u"Process Resolution", None))
        self.checkBox_enable_full_res.setText(QCoreApplication.translate("MainWindow", u"Enable Full Res", None))
        self.label_batch_size_full.setText(QCoreApplication.translate("MainWindow", u"Batch Size:", None))
        self.checkBox_enable_low_res.setText(QCoreApplication.translate("MainWindow", u"Enable Low Res", None))
        self.label_batch_size_low.setText(QCoreApplication.translate("MainWindow", u"Batch Size:", None))
        self.label_width.setText(QCoreApplication.translate("MainWindow", u"Width:", None))
        self.label_height.setText(QCoreApplication.translate("MainWindow", u"Height:", None))
        self.checkBox_dual_output.setText(QCoreApplication.translate("MainWindow", u"Dual Output Only", None))
        self.checkBox_ffmpeg.setText(QCoreApplication.translate("MainWindow", u"ffmpeg", None))
        self.lineEdit_high_batch.setText(QCoreApplication.translate("MainWindow", u"7", None))
        self.lineEdit_low_batch.setText(QCoreApplication.translate("MainWindow", u"13", None))
        self.lineEdit_low_height.setText(QCoreApplication.translate("MainWindow", u"320", None))
        self.lineEdit_low_width.setText(QCoreApplication.translate("MainWindow", u"640", None))
        self.groupBox_splatting_settings.setTitle(QCoreApplication.translate("MainWindow", u"Splatting & Output Settings", None))
        self.label_border.setText(QCoreApplication.translate("MainWindow", u"Border:", None))
        self.label_mask.setText(QCoreApplication.translate("MainWindow", u"Mask:", None))
        self.comboBox_auto_convergence.setItemText(0, QCoreApplication.translate("MainWindow", u"Off", None))

        self.label_density.setText(QCoreApplication.translate("MainWindow", u"Density:", None))
        self.label_extrusion.setText(QCoreApplication.translate("MainWindow", u"Extrusion:", None))
        self.label_auto_convergence.setText(QCoreApplication.translate("MainWindow", u"Auto-Converge:", None))
        self.label_process_length.setText(QCoreApplication.translate("MainWindow", u"Process Length:", None))
        self.comboBox_mask_type.setItemText(0, QCoreApplication.translate("MainWindow", u"M2S", None))

        self.comboBox_border.setItemText(0, QCoreApplication.translate("MainWindow", u"Auto Adv.", None))

        self.label_mesh_bias.setText(QCoreApplication.translate("MainWindow", u"Mesh Bias:", None))
        self.label_dolly.setText(QCoreApplication.translate("MainWindow", u"Dolly:", None))
        self.groupBox_depth_map_preprocessing.setTitle(QCoreApplication.translate("MainWindow", u"Depth Map Pre-processing", None))
        self.label_blur_left_value.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_blur_y_value.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_blur_y.setText(QCoreApplication.translate("MainWindow", u"Blur Y:", None))
        self.label_dilate_left.setText(QCoreApplication.translate("MainWindow", u"Dilate L:", None))
        self.label_dilate_y.setText(QCoreApplication.translate("MainWindow", u"Dilate Y:", None))
        self.label_blur_left.setText(QCoreApplication.translate("MainWindow", u"Blur L:", None))
        self.label_dilate_x.setText(QCoreApplication.translate("MainWindow", u"Dilate X:", None))
        self.label_blur_x.setText(QCoreApplication.translate("MainWindow", u"Blur X:", None))
        self.label_blue_x_value.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.label_dilate_y_value.setText(QCoreApplication.translate("MainWindow", u"3.5", None))
        self.lineEdit_blur_bias.setText(QCoreApplication.translate("MainWindow", u"0.5", None))
        self.label_dilate_left_value.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.label_dilate_x_value.setText(QCoreApplication.translate("MainWindow", u"7", None))
        self.groupBox_stereo_projection.setTitle(QCoreApplication.translate("MainWindow", u"Stereo Projection", None))
        self.label_gamma.setText(QCoreApplication.translate("MainWindow", u"Gamma:", None))
        self.label_gamma_value.setText(QCoreApplication.translate("MainWindow", u"1.0", None))
        self.label_disparity.setText(QCoreApplication.translate("MainWindow", u"Disparity:", None))
        self.label_disparity_value.setText(QCoreApplication.translate("MainWindow", u"35", None))
        self.label_convergence.setText(QCoreApplication.translate("MainWindow", u"Convergence:", None))
        self.label_convergence_value.setText(QCoreApplication.translate("MainWindow", u"1.0", None))
        self.label_border_width.setText(QCoreApplication.translate("MainWindow", u"Border Width:", None))
        self.label_border_width_value.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.label_bias.setText(QCoreApplication.translate("MainWindow", u"Border Bias:", None))
        self.label_bias_value.setText(QCoreApplication.translate("MainWindow", u"0.5", None))
        self.checkBox_normalization.setText(QCoreApplication.translate("MainWindow", u"Global Normalization", None))
        self.checkBox_resume.setText(QCoreApplication.translate("MainWindow", u"Resume", None))
        self.checkBox_cross_view.setText(QCoreApplication.translate("MainWindow", u"Cross View", None))
#if QT_CONFIG(shortcut)
        self.checkBox_cross_view.setShortcut(QCoreApplication.translate("MainWindow", u"X", None))
#endif // QT_CONFIG(shortcut)
        self.groupBox_current_processing_info.setTitle(QCoreApplication.translate("MainWindow", u"Current Processing Information", None))
        self.label_info_gamma_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_info_map_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_info_frames_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_info_disparity_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_gamma_info.setText(QCoreApplication.translate("MainWindow", u"Gamma:", None))
        self.label_convergence_info.setText(QCoreApplication.translate("MainWindow", u"Converge:", None))
        self.label_filename_info.setText(QCoreApplication.translate("MainWindow", u"Filename:", None))
        self.label_resolution_info.setText(QCoreApplication.translate("MainWindow", u"Resolution:", None))
        self.label_task_info.setText(QCoreApplication.translate("MainWindow", u"Task:", None))
        self.label_frames_info.setText(QCoreApplication.translate("MainWindow", u"Frames:", None))
        self.label_info_filename_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_info_convergence_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_info_resolution_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.label_disparity_info.setText(QCoreApplication.translate("MainWindow", u"Disparity:", None))
        self.label_map_info.setText(QCoreApplication.translate("MainWindow", u"Map:", None))
        self.label_info_task_value.setText(QCoreApplication.translate("MainWindow", u"N/A", None))
        self.groupBox_dev_tools.setTitle(QCoreApplication.translate("MainWindow", u"Dev Tools", None))
        self.checkBox_crosshair.setText(QCoreApplication.translate("MainWindow", u"Crosshair", None))
        self.checkBox_dp.setText(QCoreApplication.translate("MainWindow", u"D/P", None))
        self.checkBox_multi.setText(QCoreApplication.translate("MainWindow", u"Multi", None))
        self.checkBox_splat_test.setText(QCoreApplication.translate("MainWindow", u"Splat Test", None))
        self.checkBox_white.setText(QCoreApplication.translate("MainWindow", u"White", None))
        self.checkBox_true_max.setText(QCoreApplication.translate("MainWindow", u"True Max", None))
        self.checkBox_map_test.setText(QCoreApplication.translate("MainWindow", u"Map Test", None))
        self.checkBox_skip_low_res.setText(QCoreApplication.translate("MainWindow", u"Skip Low-Res Pre-proc", None))
        self.progressBar.setFormat(QCoreApplication.translate("MainWindow", u"Progress:", None))
        self.label_status.setText(QCoreApplication.translate("MainWindow", u"Ready", None))
        self.pushButton_single.setText(QCoreApplication.translate("MainWindow", u"SINGLE", None))
        self.pushButton_start.setText(QCoreApplication.translate("MainWindow", u"START", None))
        self.label_from.setText(QCoreApplication.translate("MainWindow", u"From:", None))
        self.label_to.setText(QCoreApplication.translate("MainWindow", u"To:", None))
        self.pushButton_stop.setText(QCoreApplication.translate("MainWindow", u"STOP", None))
        self.pushButton_preview_auto_converge.setText(QCoreApplication.translate("MainWindow", u"Preview Auto-Converge", None))
        self.pushButton_auto_pass.setText(QCoreApplication.translate("MainWindow", u"AUTO-PASS", None))
        self.pushButton_update_sidecar.setText(QCoreApplication.translate("MainWindow", u"Update Sidecar", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuOptions.setTitle(QCoreApplication.translate("MainWindow", u"Options", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi

