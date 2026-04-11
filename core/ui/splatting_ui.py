# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'splatting.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    QTime,
    QUrl,
    Qt,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1333, 773)
        icon = QIcon()
        icon.addFile("core/ui/icons/logo.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet(
            "\n"
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
            "/* Tab Widget - Make it match the theme */\n"
            "QTabWidget::pane {\n"
            "    border: 1px solid palette(mid);\n"
            "    background-color: palette(window);  /* Matches your main window */\n"
            "    border-radius: 3px;\n"
            "}\n"
            "\n"
            "QTabWidget::tab-bar {\n"
            "    alignment: left;\n"
            "}\n"
            "\n"
            "QTabBar::tab {\n"
            "    background-color: palette(window);\n"
            "    color: palette(button-text);\n"
            "    padding: 6px 12px;\n"
            "    margin-right: 2px;\n"
            "    border-top-left-radius: 3px;\n"
            "    border-top-right-radius: 3px;\n"
            "}\n"
            "\n"
            "QTabBar::tab:selected {\n"
            "    background-color: palette(button);  /* Blends with the pane */\n"
            "    color: palette(text);\n"
            "    border-botto"
            "m: 1px solid palette(window);  /* Hides the separator line */\n"
            "}\n"
            "\n"
            "QTabBar::tab:hover:!selected {\n"
            "    background-color: palette(light);\n"
            "}"
        )
        self.action_load_settings = QAction(MainWindow)
        self.action_load_settings.setObjectName("action_load_settings")
        self.action_save = QAction(MainWindow)
        self.action_save.setObjectName("action_save")
        self.action_save_to_file = QAction(MainWindow)
        self.action_save_to_file.setObjectName("action_save_to_file")
        self.action_load_fsexport = QAction(MainWindow)
        self.action_load_fsexport.setObjectName("action_load_fsexport")
        self.action_fsexport_to_sidecar = QAction(MainWindow)
        self.action_fsexport_to_sidecar.setObjectName("action_fsexport_to_sidecar")
        self.action_restore_from_finished = QAction(MainWindow)
        self.action_restore_from_finished.setObjectName("action_restore_from_finished")
        self.action_encoder = QAction(MainWindow)
        self.action_encoder.setObjectName("action_encoder")
        self.action_update_from_sidecar = QAction(MainWindow)
        self.action_update_from_sidecar.setObjectName("action_update_from_sidecar")
        self.action_update_from_sidecar.setCheckable(True)
        self.action_auto_update_sidecar = QAction(MainWindow)
        self.action_auto_update_sidecar.setObjectName("action_auto_update_sidecar")
        self.action_auto_update_sidecar.setCheckable(True)
        self.action_guide = QAction(MainWindow)
        self.action_guide.setObjectName("action_guide")
        self.action_calculator = QAction(MainWindow)
        self.action_calculator.setObjectName("action_calculator")
        self.action_debug = QAction(MainWindow)
        self.action_debug.setObjectName("action_debug")
        self.action_debug.setCheckable(True)
        self.action_about = QAction(MainWindow)
        self.action_about.setObjectName("action_about")
        self.action_exit = QAction(MainWindow)
        self.action_exit.setObjectName("action_exit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.gridLayout_1 = QGridLayout()
        self.gridLayout_1.setObjectName("gridLayout_1")
        self.groupBox_input_output = QGroupBox(self.centralwidget)
        self.groupBox_input_output.setObjectName("groupBox_input_output")
        self.groupBox_input_output.setMaximumSize(QSize(16777215, 160))
        self.gridLayout_2 = QGridLayout(self.groupBox_input_output)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_browse_output = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_output.setObjectName("pushButton_browse_output")
        self.pushButton_browse_output.setAcceptDrops(True)

        self.gridLayout_2.addWidget(self.pushButton_browse_output, 2, 2, 1, 1)

        self.label_input_depth = QLabel(self.groupBox_input_output)
        self.label_input_depth.setObjectName("label_input_depth")
        self.label_input_depth.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_2.addWidget(self.label_input_depth, 1, 0, 1, 1)

        self.pushButton_browse_depth = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_depth.setObjectName("pushButton_browse_depth")
        self.pushButton_browse_depth.setAcceptDrops(True)

        self.gridLayout_2.addWidget(self.pushButton_browse_depth, 1, 2, 1, 1)

        self.lineEdit_input_source = QLineEdit(self.groupBox_input_output)
        self.lineEdit_input_source.setObjectName("lineEdit_input_source")

        self.gridLayout_2.addWidget(self.lineEdit_input_source, 0, 1, 1, 1)

        self.pushButton_select_depth = QPushButton(self.groupBox_input_output)
        self.pushButton_select_depth.setObjectName("pushButton_select_depth")
        self.pushButton_select_depth.setAcceptDrops(True)

        self.gridLayout_2.addWidget(self.pushButton_select_depth, 1, 3, 1, 1)

        self.checkBox_multi_map = QCheckBox(self.groupBox_input_output)
        self.checkBox_multi_map.setObjectName("checkBox_multi_map")

        self.gridLayout_2.addWidget(self.checkBox_multi_map, 2, 3, 1, 1)

        self.lineEdit_output_splatted = QLineEdit(self.groupBox_input_output)
        self.lineEdit_output_splatted.setObjectName("lineEdit_output_splatted")

        self.gridLayout_2.addWidget(self.lineEdit_output_splatted, 2, 1, 1, 1)

        self.label_input_source = QLabel(self.groupBox_input_output)
        self.label_input_source.setObjectName("label_input_source")
        self.label_input_source.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_2.addWidget(self.label_input_source, 0, 0, 1, 1)

        self.label_output_splatted = QLabel(self.groupBox_input_output)
        self.label_output_splatted.setObjectName("label_output_splatted")
        self.label_output_splatted.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_2.addWidget(self.label_output_splatted, 2, 0, 1, 1)

        self.pushButton_select_source = QPushButton(self.groupBox_input_output)
        self.pushButton_select_source.setObjectName("pushButton_select_source")
        self.pushButton_select_source.setAcceptDrops(True)

        self.gridLayout_2.addWidget(self.pushButton_select_source, 0, 3, 1, 1)

        self.pushButton_browse_source = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_source.setObjectName("pushButton_browse_source")
        self.pushButton_browse_source.setAcceptDrops(True)

        self.gridLayout_2.addWidget(self.pushButton_browse_source, 0, 2, 1, 1)

        self.lineEdit_input_depth = QLineEdit(self.groupBox_input_output)
        self.lineEdit_input_depth.setObjectName("lineEdit_input_depth")

        self.gridLayout_2.addWidget(self.lineEdit_input_depth, 1, 1, 1, 1)

        self.pushButton_browse_sidecar = QPushButton(self.groupBox_input_output)
        self.pushButton_browse_sidecar.setObjectName("pushButton_browse_sidecar")
        self.pushButton_browse_sidecar.setAcceptDrops(True)

        self.gridLayout_2.addWidget(self.pushButton_browse_sidecar, 3, 2, 1, 1)

        self.lineEdit_inout_sidecar = QLineEdit(self.groupBox_input_output)
        self.lineEdit_inout_sidecar.setObjectName("lineEdit_inout_sidecar")

        self.gridLayout_2.addWidget(self.lineEdit_inout_sidecar, 3, 1, 1, 1)

        self.label_output_sidecar = QLabel(self.groupBox_input_output)
        self.label_output_sidecar.setObjectName("label_output_sidecar")
        self.label_output_sidecar.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_2.addWidget(self.label_output_sidecar, 3, 0, 1, 1)

        self.gridLayout_1.addWidget(self.groupBox_input_output, 0, 0, 1, 1)

        self.gridLayout_9.addLayout(self.gridLayout_1, 0, 0, 1, 1)

        self.line_1 = QFrame(self.centralwidget)
        self.line_1.setObjectName("line_1")
        self.line_1.setFrameShape(QFrame.Shape.HLine)
        self.line_1.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout_9.addWidget(self.line_1, 1, 0, 1, 1)

        self.horizontalLayout_frame_info = QHBoxLayout()
        self.horizontalLayout_frame_info.setObjectName("horizontalLayout_frame_info")
        self.label_frame_info = QLabel(self.centralwidget)
        self.label_frame_info.setObjectName("label_frame_info")

        self.horizontalLayout_frame_info.addWidget(self.label_frame_info)

        self.horizontalSlider = QSlider(self.centralwidget)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setMinimumSize(QSize(0, 60))
        self.horizontalSlider.setStyleSheet(
            "\n"
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
            "    }"
        )
        self.horizontalSlider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_frame_info.addWidget(self.horizontalSlider)

        self.gridLayout_9.addLayout(self.horizontalLayout_frame_info, 2, 0, 1, 1)

        self.horizontalLayout_preview_controls = QHBoxLayout()
        self.horizontalLayout_preview_controls.setObjectName("horizontalLayout_preview_controls")
        self.label_preview_source = QLabel(self.centralwidget)
        self.label_preview_source.setObjectName("label_preview_source")

        self.horizontalLayout_preview_controls.addWidget(self.label_preview_source)

        self.comboBox_preview_source = QComboBox(self.centralwidget)
        self.comboBox_preview_source.addItem("")
        self.comboBox_preview_source.setObjectName("comboBox_preview_source")
        self.comboBox_preview_source.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_preview_controls.addWidget(self.comboBox_preview_source)

        self.pushButton_load_refresh = QPushButton(self.centralwidget)
        self.pushButton_load_refresh.setObjectName("pushButton_load_refresh")
        self.pushButton_load_refresh.setMinimumSize(QSize(140, 0))
        self.pushButton_load_refresh.setMaximumSize(QSize(140, 16777215))

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_load_refresh)

        self.horizontalSpacer_2 = QSpacerItem(20, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_preview_controls.addItem(self.horizontalSpacer_2)

        self.pushButton_prev = QPushButton(self.centralwidget)
        self.pushButton_prev.setObjectName("pushButton_prev")

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_prev)

        self.pushButton_next = QPushButton(self.centralwidget)
        self.pushButton_next.setObjectName("pushButton_next")

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_next)

        self.label_jump_to = QLabel(self.centralwidget)
        self.label_jump_to.setObjectName("label_jump_to")

        self.horizontalLayout_preview_controls.addWidget(self.label_jump_to)

        self.lineEdit_jump_to = QLineEdit(self.centralwidget)
        self.lineEdit_jump_to.setObjectName("lineEdit_jump_to")
        self.lineEdit_jump_to.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_preview_controls.addWidget(self.lineEdit_jump_to)

        self.label_video_info = QLabel(self.centralwidget)
        self.label_video_info.setObjectName("label_video_info")

        self.horizontalLayout_preview_controls.addWidget(self.label_video_info)

        self.pushButton_play = QPushButton(self.centralwidget)
        self.pushButton_play.setObjectName("pushButton_play")
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaybackStart))
        self.pushButton_play.setIcon(icon1)

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_play)

        self.pushButton_fast_forward = QPushButton(self.centralwidget)
        self.pushButton_fast_forward.setObjectName("pushButton_fast_forward")
        icon2 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaSeekForward))
        self.pushButton_fast_forward.setIcon(icon2)

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_fast_forward)

        self.spinBox_ff_speed = QSpinBox(self.centralwidget)
        self.spinBox_ff_speed.setObjectName("spinBox_ff_speed")
        self.spinBox_ff_speed.setMaximumSize(QSize(90, 16777215))
        self.spinBox_ff_speed.setValue(5)

        self.horizontalLayout_preview_controls.addWidget(self.spinBox_ff_speed)

        self.pushButton_loop_toggle = QPushButton(self.centralwidget)
        self.pushButton_loop_toggle.setObjectName("pushButton_loop_toggle")
        icon3 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.MediaPlaylistRepeat))
        self.pushButton_loop_toggle.setIcon(icon3)
        self.pushButton_loop_toggle.setCheckable(True)
        self.pushButton_loop_toggle.setChecked(False)
        self.pushButton_loop_toggle.setFlat(False)

        self.horizontalLayout_preview_controls.addWidget(self.pushButton_loop_toggle)

        self.label_depth_map = QLabel(self.centralwidget)
        self.label_depth_map.setObjectName("label_depth_map")
        self.label_depth_map.setMinimumSize(QSize(70, 0))
        self.label_depth_map.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.horizontalLayout_preview_controls.addWidget(self.label_depth_map)

        self.comboBox_map_select = QComboBox(self.centralwidget)
        self.comboBox_map_select.addItem("")
        self.comboBox_map_select.addItem("")
        self.comboBox_map_select.setObjectName("comboBox_map_select")
        self.comboBox_map_select.setEnabled(False)

        self.horizontalLayout_preview_controls.addWidget(self.comboBox_map_select)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_preview_controls.addItem(self.horizontalSpacer_3)

        self.label_preview_scale = QLabel(self.centralwidget)
        self.label_preview_scale.setObjectName("label_preview_scale")

        self.horizontalLayout_preview_controls.addWidget(self.label_preview_scale)

        self.comboBox_preview_scale = QComboBox(self.centralwidget)
        self.comboBox_preview_scale.addItem("")
        self.comboBox_preview_scale.setObjectName("comboBox_preview_scale")
        self.comboBox_preview_scale.setMinimumSize(QSize(60, 0))

        self.horizontalLayout_preview_controls.addWidget(self.comboBox_preview_scale)

        self.gridLayout_9.addLayout(self.horizontalLayout_preview_controls, 3, 0, 1, 1)

        self.horizontalLayout_main_content = QHBoxLayout()
        self.horizontalLayout_main_content.setObjectName("horizontalLayout_main_content")
        self.verticalLayout_left_panel = QVBoxLayout()
        self.verticalLayout_left_panel.setObjectName("verticalLayout_left_panel")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_output_settings = QWidget()
        self.tab_output_settings.setObjectName("tab_output_settings")
        self.verticalLayout = QVBoxLayout(self.tab_output_settings)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.groupBox_splatting_settings = QGroupBox(self.tab_output_settings)
        self.groupBox_splatting_settings.setObjectName("groupBox_splatting_settings")
        self.groupBox_splatting_settings.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_4 = QGridLayout(self.groupBox_splatting_settings)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lineEdit_process_length = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_process_length.setObjectName("lineEdit_process_length")

        self.gridLayout_4.addWidget(self.lineEdit_process_length, 0, 1, 1, 1)

        self.label_mesh_bias = QLabel(self.groupBox_splatting_settings)
        self.label_mesh_bias.setObjectName("label_mesh_bias")
        self.label_mesh_bias.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_mesh_bias, 2, 0, 1, 1)

        self.label_border = QLabel(self.groupBox_splatting_settings)
        self.label_border.setObjectName("label_border")
        self.label_border.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_border, 1, 2, 1, 1)

        self.comboBox_border = QComboBox(self.groupBox_splatting_settings)
        self.comboBox_border.addItem("")
        self.comboBox_border.setObjectName("comboBox_border")

        self.gridLayout_4.addWidget(self.comboBox_border, 1, 3, 1, 1)

        self.lineEdit_mesh_bias = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_bias.setObjectName("lineEdit_mesh_bias")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_bias, 2, 1, 1, 1)

        self.label_extrusion = QLabel(self.groupBox_splatting_settings)
        self.label_extrusion.setObjectName("label_extrusion")
        self.label_extrusion.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_extrusion, 3, 0, 1, 1)

        self.label_process_length = QLabel(self.groupBox_splatting_settings)
        self.label_process_length.setObjectName("label_process_length")
        self.label_process_length.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_process_length, 0, 0, 1, 1)

        self.label_auto_convergence = QLabel(self.groupBox_splatting_settings)
        self.label_auto_convergence.setObjectName("label_auto_convergence")
        self.label_auto_convergence.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_auto_convergence, 0, 2, 1, 1)

        self.lineEdit_mesh_extrusion = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_extrusion.setObjectName("lineEdit_mesh_extrusion")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_extrusion, 3, 1, 1, 1)

        self.comboBox_mask_type = QComboBox(self.groupBox_splatting_settings)
        self.comboBox_mask_type.addItem("")
        self.comboBox_mask_type.setObjectName("comboBox_mask_type")

        self.gridLayout_4.addWidget(self.comboBox_mask_type, 1, 1, 1, 1)

        self.label_dolly = QLabel(self.groupBox_splatting_settings)
        self.label_dolly.setObjectName("label_dolly")
        self.label_dolly.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_dolly, 2, 2, 1, 1)

        self.label_density = QLabel(self.groupBox_splatting_settings)
        self.label_density.setObjectName("label_density")
        self.label_density.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_density, 3, 2, 1, 1)

        self.lineEdit_mesh_dolly = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_dolly.setObjectName("lineEdit_mesh_dolly")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_dolly, 2, 3, 1, 1)

        self.label_mask = QLabel(self.groupBox_splatting_settings)
        self.label_mask.setObjectName("label_mask")
        self.label_mask.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_4.addWidget(self.label_mask, 1, 0, 1, 1)

        self.comboBox_auto_convergence = QComboBox(self.groupBox_splatting_settings)
        self.comboBox_auto_convergence.addItem("")
        self.comboBox_auto_convergence.setObjectName("comboBox_auto_convergence")

        self.gridLayout_4.addWidget(self.comboBox_auto_convergence, 0, 3, 1, 1)

        self.lineEdit_mesh_density = QLineEdit(self.groupBox_splatting_settings)
        self.lineEdit_mesh_density.setObjectName("lineEdit_mesh_density")

        self.gridLayout_4.addWidget(self.lineEdit_mesh_density, 3, 3, 1, 1)

        self.verticalLayout.addWidget(self.groupBox_splatting_settings)

        self.tabWidget.addTab(self.tab_output_settings, "")
        self.tab_output_types = QWidget()
        self.tab_output_types.setObjectName("tab_output_types")
        self.verticalLayout_2 = QVBoxLayout(self.tab_output_types)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(6, 6, 6, 6)
        self.groupBox_process_resolution = QGroupBox(self.tab_output_types)
        self.groupBox_process_resolution.setObjectName("groupBox_process_resolution")
        self.groupBox_process_resolution.setMaximumSize(QSize(16777215, 16777215))
        self.gridLayout_3 = QGridLayout(self.groupBox_process_resolution)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_enable_full_res = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_enable_full_res.setObjectName("checkBox_enable_full_res")
        self.checkBox_enable_full_res.setChecked(True)

        self.gridLayout_3.addWidget(self.checkBox_enable_full_res, 0, 0, 1, 2)

        self.label_batch_size_full = QLabel(self.groupBox_process_resolution)
        self.label_batch_size_full.setObjectName("label_batch_size_full")

        self.gridLayout_3.addWidget(self.label_batch_size_full, 0, 2, 1, 1)

        self.checkBox_enable_low_res = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_enable_low_res.setObjectName("checkBox_enable_low_res")
        self.checkBox_enable_low_res.setChecked(True)

        self.gridLayout_3.addWidget(self.checkBox_enable_low_res, 1, 0, 1, 2)

        self.label_batch_size_low = QLabel(self.groupBox_process_resolution)
        self.label_batch_size_low.setObjectName("label_batch_size_low")

        self.gridLayout_3.addWidget(self.label_batch_size_low, 1, 2, 1, 1)

        self.label_width = QLabel(self.groupBox_process_resolution)
        self.label_width.setObjectName("label_width")

        self.gridLayout_3.addWidget(self.label_width, 2, 0, 1, 1)

        self.label_height = QLabel(self.groupBox_process_resolution)
        self.label_height.setObjectName("label_height")

        self.gridLayout_3.addWidget(self.label_height, 2, 2, 1, 1)

        self.checkBox_dual_output = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_dual_output.setObjectName("checkBox_dual_output")

        self.gridLayout_3.addWidget(self.checkBox_dual_output, 0, 4, 1, 1)

        self.checkBox_ffmpeg = QCheckBox(self.groupBox_process_resolution)
        self.checkBox_ffmpeg.setObjectName("checkBox_ffmpeg")

        self.gridLayout_3.addWidget(self.checkBox_ffmpeg, 1, 4, 1, 1)

        self.lineEdit_high_batch = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_high_batch.setObjectName("lineEdit_high_batch")

        self.gridLayout_3.addWidget(self.lineEdit_high_batch, 0, 3, 1, 1)

        self.lineEdit_low_batch = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_low_batch.setObjectName("lineEdit_low_batch")

        self.gridLayout_3.addWidget(self.lineEdit_low_batch, 1, 3, 1, 1)

        self.lineEdit_low_height = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_low_height.setObjectName("lineEdit_low_height")

        self.gridLayout_3.addWidget(self.lineEdit_low_height, 2, 3, 1, 1)

        self.lineEdit_low_width = QLineEdit(self.groupBox_process_resolution)
        self.lineEdit_low_width.setObjectName("lineEdit_low_width")

        self.gridLayout_3.addWidget(self.lineEdit_low_width, 2, 1, 1, 1)

        self.verticalLayout_2.addWidget(self.groupBox_process_resolution)

        self.groupBox_individual_outputs = QGroupBox(self.tab_output_types)
        self.groupBox_individual_outputs.setObjectName("groupBox_individual_outputs")
        self.gridLayout = QGridLayout(self.groupBox_individual_outputs)
        self.gridLayout.setObjectName("gridLayout")
        self.checkBox_splat_low = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_splat_low.setObjectName("checkBox_splat_low")

        self.gridLayout.addWidget(self.checkBox_splat_low, 0, 0, 1, 1)

        self.checkBox_splat_hi = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_splat_hi.setObjectName("checkBox_splat_hi")

        self.gridLayout.addWidget(self.checkBox_splat_hi, 0, 1, 1, 1)

        self.checkBox_anaglyph = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_anaglyph.setObjectName("checkBox_anaglyph")

        self.gridLayout.addWidget(self.checkBox_anaglyph, 0, 2, 1, 1)

        self.checkBox = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox.setObjectName("checkBox")

        self.gridLayout.addWidget(self.checkBox, 1, 0, 1, 1)

        self.checkBox_mask = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_mask.setObjectName("checkBox_mask")

        self.gridLayout.addWidget(self.checkBox_mask, 1, 1, 1, 1)

        self.checkBox_flowmap_x = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_flowmap_x.setObjectName("checkBox_flowmap_x")
        self.checkBox_flowmap_x.setEnabled(False)

        self.gridLayout.addWidget(self.checkBox_flowmap_x, 1, 2, 1, 1)

        self.checkBox_mesh_sbs = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_mesh_sbs.setObjectName("checkBox_mesh_sbs")

        self.gridLayout.addWidget(self.checkBox_mesh_sbs, 2, 0, 1, 1)

        self.checkBox_splat_sbs = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_splat_sbs.setObjectName("checkBox_splat_sbs")

        self.gridLayout.addWidget(self.checkBox_splat_sbs, 2, 1, 1, 1)

        self.checkBox_flowmap_full = QCheckBox(self.groupBox_individual_outputs)
        self.checkBox_flowmap_full.setObjectName("checkBox_flowmap_full")
        self.checkBox_flowmap_full.setEnabled(False)

        self.gridLayout.addWidget(self.checkBox_flowmap_full, 2, 2, 1, 1)

        self.verticalLayout_2.addWidget(self.groupBox_individual_outputs)

        self.tabWidget.addTab(self.tab_output_types, "")

        self.verticalLayout_left_panel.addWidget(self.tabWidget)

        self.horizontalLayout_main_content.addLayout(self.verticalLayout_left_panel)

        self.verticalLayout_right_panel = QVBoxLayout()
        self.verticalLayout_right_panel.setObjectName("verticalLayout_right_panel")
        self.groupBox_depth_map_preprocessing = QGroupBox(self.centralwidget)
        self.groupBox_depth_map_preprocessing.setObjectName("groupBox_depth_map_preprocessing")
        self.gridLayout_5 = QGridLayout(self.groupBox_depth_map_preprocessing)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.gridLayout_5.setContentsMargins(-1, 6, -1, -1)
        self.label_blur_left_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_left_value.setObjectName("label_blur_left_value")
        self.label_blur_left_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_blur_left_value, 2, 5, 1, 1)

        self.horizontalSlider_blur_y = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_blur_y.setObjectName("horizontalSlider_blur_y")
        self.horizontalSlider_blur_y.setValue(5)
        self.horizontalSlider_blur_y.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_blur_y, 1, 4, 1, 1)

        self.label_blur_y_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_y_value.setObjectName("label_blur_y_value")
        self.label_blur_y_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_blur_y_value, 1, 5, 1, 1)

        self.horizontalSlider_blur_left = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_blur_left.setObjectName("horizontalSlider_blur_left")
        self.horizontalSlider_blur_left.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_blur_left, 2, 4, 1, 1)

        self.label_blur_y = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_y.setObjectName("label_blur_y")
        self.label_blur_y.setMinimumSize(QSize(44, 0))
        self.label_blur_y.setMaximumSize(QSize(40, 16777215))
        self.label_blur_y.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_5.addWidget(self.label_blur_y, 1, 3, 1, 1)

        self.label_dilate_left = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_left.setObjectName("label_dilate_left")
        self.label_dilate_left.setMinimumSize(QSize(44, 0))
        self.label_dilate_left.setMaximumSize(QSize(40, 16777215))
        self.label_dilate_left.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_5.addWidget(self.label_dilate_left, 2, 0, 1, 1)

        self.horizontalSlider_dilate_left = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_dilate_left.setObjectName("horizontalSlider_dilate_left")
        self.horizontalSlider_dilate_left.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_dilate_left, 2, 1, 1, 1)

        self.label_dilate_y = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_y.setObjectName("label_dilate_y")
        self.label_dilate_y.setMinimumSize(QSize(44, 0))
        self.label_dilate_y.setMaximumSize(QSize(40, 16777215))
        self.label_dilate_y.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_5.addWidget(self.label_dilate_y, 0, 3, 1, 1)

        self.label_blur_left = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_left.setObjectName("label_blur_left")
        self.label_blur_left.setMinimumSize(QSize(44, 0))
        self.label_blur_left.setMaximumSize(QSize(40, 16777215))
        self.label_blur_left.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_5.addWidget(self.label_blur_left, 2, 3, 1, 1)

        self.label_dilate_x = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_x.setObjectName("label_dilate_x")
        self.label_dilate_x.setMinimumSize(QSize(44, 0))
        self.label_dilate_x.setMaximumSize(QSize(40, 16777215))
        self.label_dilate_x.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_5.addWidget(self.label_dilate_x, 0, 0, 1, 1)

        self.label_blur_x = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blur_x.setObjectName("label_blur_x")
        self.label_blur_x.setMinimumSize(QSize(44, 0))
        self.label_blur_x.setMaximumSize(QSize(40, 16777215))
        self.label_blur_x.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_5.addWidget(self.label_blur_x, 1, 0, 1, 1)

        self.label_blue_x_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_blue_x_value.setObjectName("label_blue_x_value")
        self.label_blue_x_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_blue_x_value, 1, 2, 1, 1)

        self.label_dilate_y_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_y_value.setObjectName("label_dilate_y_value")
        self.label_dilate_y_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_dilate_y_value, 0, 5, 1, 1)

        self.lineEdit_blur_bias = QLineEdit(self.groupBox_depth_map_preprocessing)
        self.lineEdit_blur_bias.setObjectName("lineEdit_blur_bias")
        self.lineEdit_blur_bias.setMaximumSize(QSize(30, 16777215))

        self.gridLayout_5.addWidget(self.lineEdit_blur_bias, 2, 6, 1, 1)

        self.horizontalSlider_blur_x = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_blur_x.setObjectName("horizontalSlider_blur_x")
        self.horizontalSlider_blur_x.setValue(5)
        self.horizontalSlider_blur_x.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_blur_x, 1, 1, 1, 1)

        self.horizontalSlider_dilate_x = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_dilate_x.setObjectName("horizontalSlider_dilate_x")
        self.horizontalSlider_dilate_x.setValue(12)
        self.horizontalSlider_dilate_x.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_dilate_x, 0, 1, 1, 1)

        self.horizontalSlider_dilate_y = QSlider(self.groupBox_depth_map_preprocessing)
        self.horizontalSlider_dilate_y.setObjectName("horizontalSlider_dilate_y")
        self.horizontalSlider_dilate_y.setValue(3)
        self.horizontalSlider_dilate_y.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_5.addWidget(self.horizontalSlider_dilate_y, 0, 4, 1, 1)

        self.label_dilate_left_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_left_value.setObjectName("label_dilate_left_value")
        self.label_dilate_left_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_dilate_left_value, 2, 2, 1, 1)

        self.label_dilate_x_value = QLabel(self.groupBox_depth_map_preprocessing)
        self.label_dilate_x_value.setObjectName("label_dilate_x_value")
        self.label_dilate_x_value.setMaximumSize(QSize(16, 16777215))

        self.gridLayout_5.addWidget(self.label_dilate_x_value, 0, 2, 1, 1)

        self.gridLayout_5.setColumnStretch(0, 1)

        self.verticalLayout_right_panel.addWidget(self.groupBox_depth_map_preprocessing)

        self.groupBox_stereo_projection = QGroupBox(self.centralwidget)
        self.groupBox_stereo_projection.setObjectName("groupBox_stereo_projection")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_stereo_projection.sizePolicy().hasHeightForWidth())
        self.groupBox_stereo_projection.setSizePolicy(sizePolicy)
        self.gridLayout_8 = QGridLayout(self.groupBox_stereo_projection)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_gamma = QLabel(self.groupBox_stereo_projection)
        self.label_gamma.setObjectName("label_gamma")
        self.label_gamma.setMinimumSize(QSize(80, 0))
        self.label_gamma.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_8.addWidget(self.label_gamma, 0, 0, 1, 1)

        self.horizontalSlider_gamma = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_gamma.setObjectName("horizontalSlider_gamma")
        self.horizontalSlider_gamma.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_gamma.setValue(99)
        self.horizontalSlider_gamma.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_gamma, 0, 1, 1, 1)

        self.label_gamma_value = QLabel(self.groupBox_stereo_projection)
        self.label_gamma_value.setObjectName("label_gamma_value")
        self.label_gamma_value.setMinimumSize(QSize(20, 0))

        self.gridLayout_8.addWidget(self.label_gamma_value, 0, 2, 1, 1)

        self.label_disparity = QLabel(self.groupBox_stereo_projection)
        self.label_disparity.setObjectName("label_disparity")
        self.label_disparity.setMinimumSize(QSize(70, 0))
        self.label_disparity.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_8.addWidget(self.label_disparity, 0, 3, 1, 1)

        self.horizontalSlider_disparity = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_disparity.setObjectName("horizontalSlider_disparity")
        self.horizontalSlider_disparity.setMinimumSize(QSize(96, 0))
        self.horizontalSlider_disparity.setValue(45)
        self.horizontalSlider_disparity.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_disparity, 0, 4, 1, 1)

        self.label_disparity_value = QLabel(self.groupBox_stereo_projection)
        self.label_disparity_value.setObjectName("label_disparity_value")
        self.label_disparity_value.setMinimumSize(QSize(20, 0))

        self.gridLayout_8.addWidget(self.label_disparity_value, 0, 5, 1, 1)

        self.label_convergence = QLabel(self.groupBox_stereo_projection)
        self.label_convergence.setObjectName("label_convergence")
        self.label_convergence.setMinimumSize(QSize(80, 0))
        self.label_convergence.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_8.addWidget(self.label_convergence, 1, 0, 1, 1)

        self.horizontalSlider_convergence = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_convergence.setObjectName("horizontalSlider_convergence")
        self.horizontalSlider_convergence.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_convergence.setValue(41)
        self.horizontalSlider_convergence.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_convergence, 1, 1, 1, 4)

        self.label_convergence_value = QLabel(self.groupBox_stereo_projection)
        self.label_convergence_value.setObjectName("label_convergence_value")
        self.label_convergence_value.setMinimumSize(QSize(20, 0))

        self.gridLayout_8.addWidget(self.label_convergence_value, 1, 5, 1, 1)

        self.label_border_width = QLabel(self.groupBox_stereo_projection)
        self.label_border_width.setObjectName("label_border_width")
        self.label_border_width.setMinimumSize(QSize(80, 0))
        self.label_border_width.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_8.addWidget(self.label_border_width, 2, 0, 1, 1)

        self.horizontalSlider_border_width = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_border_width.setObjectName("horizontalSlider_border_width")
        self.horizontalSlider_border_width.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_border_width.setValue(99)
        self.horizontalSlider_border_width.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_border_width, 2, 1, 1, 1)

        self.label_border_width_value = QLabel(self.groupBox_stereo_projection)
        self.label_border_width_value.setObjectName("label_border_width_value")
        self.label_border_width_value.setMinimumSize(QSize(20, 0))

        self.gridLayout_8.addWidget(self.label_border_width_value, 2, 2, 1, 1)

        self.label_bias = QLabel(self.groupBox_stereo_projection)
        self.label_bias.setObjectName("label_bias")
        self.label_bias.setMinimumSize(QSize(70, 0))
        self.label_bias.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing | Qt.AlignmentFlag.AlignVCenter
        )

        self.gridLayout_8.addWidget(self.label_bias, 2, 3, 1, 1)

        self.horizontalSlider_border_bias = QSlider(self.groupBox_stereo_projection)
        self.horizontalSlider_border_bias.setObjectName("horizontalSlider_border_bias")
        self.horizontalSlider_border_bias.setMinimumSize(QSize(0, 0))
        self.horizontalSlider_border_bias.setValue(27)
        self.horizontalSlider_border_bias.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout_8.addWidget(self.horizontalSlider_border_bias, 2, 4, 1, 1)

        self.label_bias_value = QLabel(self.groupBox_stereo_projection)
        self.label_bias_value.setObjectName("label_bias_value")
        self.label_bias_value.setMinimumSize(QSize(20, 0))

        self.gridLayout_8.addWidget(self.label_bias_value, 2, 5, 1, 1)

        self.checkBox_normalization = QCheckBox(self.groupBox_stereo_projection)
        self.checkBox_normalization.setObjectName("checkBox_normalization")
        self.checkBox_normalization.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.checkBox_normalization, 3, 1, 1, 1)

        self.checkBox_resume = QCheckBox(self.groupBox_stereo_projection)
        self.checkBox_resume.setObjectName("checkBox_resume")
        self.checkBox_resume.setMinimumSize(QSize(0, 0))

        self.gridLayout_8.addWidget(self.checkBox_resume, 3, 3, 1, 1)

        self.checkBox_cross_view = QCheckBox(self.groupBox_stereo_projection)
        self.checkBox_cross_view.setObjectName("checkBox_cross_view")

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
        self.verticalLayout_info_panel.setObjectName("verticalLayout_info_panel")
        self.groupBox_current_processing_info = QGroupBox(self.centralwidget)
        self.groupBox_current_processing_info.setObjectName("groupBox_current_processing_info")
        self.gridLayout_7 = QGridLayout(self.groupBox_current_processing_info)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_info_gamma_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_gamma_value.setObjectName("label_info_gamma_value")

        self.gridLayout_7.addWidget(self.label_info_gamma_value, 4, 1, 1, 1)

        self.label_info_map_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_map_value.setObjectName("label_info_map_value")

        self.gridLayout_7.addWidget(self.label_info_map_value, 2, 3, 1, 1)

        self.label_info_frames_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_frames_value.setObjectName("label_info_frames_value")

        self.gridLayout_7.addWidget(self.label_info_frames_value, 3, 1, 1, 1)

        self.label_info_disparity_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_disparity_value.setObjectName("label_info_disparity_value")

        self.gridLayout_7.addWidget(self.label_info_disparity_value, 0, 3, 1, 1)

        self.label_gamma_info = QLabel(self.groupBox_current_processing_info)
        self.label_gamma_info.setObjectName("label_gamma_info")

        self.gridLayout_7.addWidget(self.label_gamma_info, 4, 0, 1, 1)

        self.label_convergence_info = QLabel(self.groupBox_current_processing_info)
        self.label_convergence_info.setObjectName("label_convergence_info")

        self.gridLayout_7.addWidget(self.label_convergence_info, 1, 2, 1, 1)

        self.label_filename_info = QLabel(self.groupBox_current_processing_info)
        self.label_filename_info.setObjectName("label_filename_info")

        self.gridLayout_7.addWidget(self.label_filename_info, 0, 0, 1, 1)

        self.label_resolution_info = QLabel(self.groupBox_current_processing_info)
        self.label_resolution_info.setObjectName("label_resolution_info")

        self.gridLayout_7.addWidget(self.label_resolution_info, 2, 0, 1, 1)

        self.label_task_info = QLabel(self.groupBox_current_processing_info)
        self.label_task_info.setObjectName("label_task_info")

        self.gridLayout_7.addWidget(self.label_task_info, 1, 0, 1, 1)

        self.label_frames_info = QLabel(self.groupBox_current_processing_info)
        self.label_frames_info.setObjectName("label_frames_info")

        self.gridLayout_7.addWidget(self.label_frames_info, 3, 0, 1, 1)

        self.label_info_filename_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_filename_value.setObjectName("label_info_filename_value")

        self.gridLayout_7.addWidget(self.label_info_filename_value, 0, 1, 1, 1)

        self.label_info_convergence_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_convergence_value.setObjectName("label_info_convergence_value")

        self.gridLayout_7.addWidget(self.label_info_convergence_value, 1, 3, 1, 1)

        self.label_info_resolution_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_resolution_value.setObjectName("label_info_resolution_value")

        self.gridLayout_7.addWidget(self.label_info_resolution_value, 2, 1, 1, 1)

        self.label_disparity_info = QLabel(self.groupBox_current_processing_info)
        self.label_disparity_info.setObjectName("label_disparity_info")

        self.gridLayout_7.addWidget(self.label_disparity_info, 0, 2, 1, 1)

        self.label_map_info = QLabel(self.groupBox_current_processing_info)
        self.label_map_info.setObjectName("label_map_info")

        self.gridLayout_7.addWidget(self.label_map_info, 2, 2, 1, 1)

        self.label_info_task_value = QLabel(self.groupBox_current_processing_info)
        self.label_info_task_value.setObjectName("label_info_task_value")

        self.gridLayout_7.addWidget(self.label_info_task_value, 1, 1, 1, 1)

        self.gridLayout_7.setColumnStretch(0, 2)
        self.gridLayout_7.setColumnStretch(1, 4)
        self.gridLayout_7.setColumnStretch(2, 2)
        self.gridLayout_7.setColumnStretch(3, 4)

        self.verticalLayout_info_panel.addWidget(self.groupBox_current_processing_info)

        self.groupBox_dev_tools = QGroupBox(self.centralwidget)
        self.groupBox_dev_tools.setObjectName("groupBox_dev_tools")
        self.gridLayout_6 = QGridLayout(self.groupBox_dev_tools)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.checkBox_crosshair = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_crosshair.setObjectName("checkBox_crosshair")
        self.checkBox_crosshair.setEnabled(True)
        self.checkBox_crosshair.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_crosshair, 1, 0, 1, 1)

        self.checkBox_dp = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_dp.setObjectName("checkBox_dp")
        self.checkBox_dp.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_dp, 1, 3, 1, 1)

        self.checkBox_multi = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_multi.setObjectName("checkBox_multi")
        self.checkBox_multi.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_multi, 1, 2, 1, 1)

        self.checkBox_splat_test = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_splat_test.setObjectName("checkBox_splat_test")

        self.gridLayout_6.addWidget(self.checkBox_splat_test, 0, 2, 1, 1)

        self.checkBox_white = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_white.setObjectName("checkBox_white")
        self.checkBox_white.setChecked(False)

        self.gridLayout_6.addWidget(self.checkBox_white, 1, 1, 1, 1)

        self.checkBox_true_max = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_true_max.setObjectName("checkBox_true_max")

        self.gridLayout_6.addWidget(self.checkBox_true_max, 0, 3, 1, 1)

        self.checkBox_map_test = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_map_test.setObjectName("checkBox_map_test")

        self.gridLayout_6.addWidget(self.checkBox_map_test, 0, 1, 1, 1)

        self.checkBox_skip_low_res = QCheckBox(self.groupBox_dev_tools)
        self.checkBox_skip_low_res.setObjectName("checkBox_skip_low_res")

        self.gridLayout_6.addWidget(self.checkBox_skip_low_res, 0, 0, 1, 1)

        self.verticalLayout_info_panel.addWidget(self.groupBox_dev_tools)

        self.horizontalLayout_main_content.addLayout(self.verticalLayout_info_panel)

        self.horizontalLayout_main_content.setStretch(0, 1)
        self.horizontalLayout_main_content.setStretch(1, 5)
        self.horizontalLayout_main_content.setStretch(2, 2)

        self.gridLayout_9.addLayout(self.horizontalLayout_main_content, 4, 0, 1, 1)

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setValue(0)

        self.gridLayout_9.addWidget(self.progressBar, 5, 0, 1, 1)

        self.label_status = QLabel(self.centralwidget)
        self.label_status.setObjectName("label_status")
        self.label_status.setMinimumSize(QSize(0, 50))
        self.label_status.setMaximumSize(QSize(16777215, 50))
        self.label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout_9.addWidget(self.label_status, 6, 0, 1, 1)

        self.horizontalLayout_bottom_buttons = QHBoxLayout()
        self.horizontalLayout_bottom_buttons.setObjectName("horizontalLayout_bottom_buttons")
        self.pushButton_single = QPushButton(self.centralwidget)
        self.pushButton_single.setObjectName("pushButton_single")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_single)

        self.pushButton_start = QPushButton(self.centralwidget)
        self.pushButton_start.setObjectName("pushButton_start")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_start)

        self.label_from = QLabel(self.centralwidget)
        self.label_from.setObjectName("label_from")

        self.horizontalLayout_bottom_buttons.addWidget(self.label_from)

        self.spinBox_from = QSpinBox(self.centralwidget)
        self.spinBox_from.setObjectName("spinBox_from")

        self.horizontalLayout_bottom_buttons.addWidget(self.spinBox_from)

        self.label_to = QLabel(self.centralwidget)
        self.label_to.setObjectName("label_to")

        self.horizontalLayout_bottom_buttons.addWidget(self.label_to)

        self.spinBox_to = QSpinBox(self.centralwidget)
        self.spinBox_to.setObjectName("spinBox_to")

        self.horizontalLayout_bottom_buttons.addWidget(self.spinBox_to)

        self.pushButton_stop = QPushButton(self.centralwidget)
        self.pushButton_stop.setObjectName("pushButton_stop")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_stop)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_bottom_buttons.addItem(self.horizontalSpacer_4)

        self.pushButton_preview_auto_converge = QPushButton(self.centralwidget)
        self.pushButton_preview_auto_converge.setObjectName("pushButton_preview_auto_converge")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_preview_auto_converge)

        self.pushButton_auto_pass = QPushButton(self.centralwidget)
        self.pushButton_auto_pass.setObjectName("pushButton_auto_pass")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_auto_pass)

        self.pushButton_update_sidecar = QPushButton(self.centralwidget)
        self.pushButton_update_sidecar.setObjectName("pushButton_update_sidecar")

        self.horizontalLayout_bottom_buttons.addWidget(self.pushButton_update_sidecar)

        self.gridLayout_9.addLayout(self.horizontalLayout_bottom_buttons, 7, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuOptions = QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
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
        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Stereocrafter Splatting (Batch) 26-03-29.0", None)
        )
        self.action_load_settings.setText(QCoreApplication.translate("MainWindow", "Load settings from File...", None))
        self.action_save.setText(QCoreApplication.translate("MainWindow", "Save Settings", None))
        # if QT_CONFIG(shortcut)
        self.action_save.setShortcut(QCoreApplication.translate("MainWindow", "Ctrl+S", None))
        # endif // QT_CONFIG(shortcut)
        self.action_save_to_file.setText(QCoreApplication.translate("MainWindow", "Save Settings to File...", None))
        self.action_load_fsexport.setText(
            QCoreApplication.translate("MainWindow", "Load Fusion Export (,fsexport)...", None)
        )
        self.action_fsexport_to_sidecar.setText(
            QCoreApplication.translate("MainWindow", "FSExport to Custom Sidecar...", None)
        )
        self.action_restore_from_finished.setText(
            QCoreApplication.translate("MainWindow", "Restore Finished Files", None)
        )
        self.action_encoder.setText(QCoreApplication.translate("MainWindow", "Encoder Settings", None))
        self.action_update_from_sidecar.setText(QCoreApplication.translate("MainWindow", "Load Sidecar Data", None))
        self.action_auto_update_sidecar.setText(QCoreApplication.translate("MainWindow", "Save Sidecar on Next", None))
        self.action_guide.setText(QCoreApplication.translate("MainWindow", "User Guide", None))
        self.action_calculator.setText(QCoreApplication.translate("MainWindow", "VRAM Calculator", None))
        self.action_debug.setText(QCoreApplication.translate("MainWindow", "Debug Logging", None))
        self.action_about.setText(QCoreApplication.translate("MainWindow", "About Splatting GUI", None))
        self.action_exit.setText(QCoreApplication.translate("MainWindow", "Exit", None))
        self.groupBox_input_output.setTitle(QCoreApplication.translate("MainWindow", "Input/Output Folders", None))
        self.pushButton_browse_output.setText(QCoreApplication.translate("MainWindow", "Browse Folder", None))
        self.label_input_depth.setText(QCoreApplication.translate("MainWindow", "Input Depth Maps:", None))
        self.pushButton_browse_depth.setText(QCoreApplication.translate("MainWindow", "Browse Folder", None))
        self.lineEdit_input_source.setText(QCoreApplication.translate("MainWindow", "./workspace/clips", None))
        self.pushButton_select_depth.setText(QCoreApplication.translate("MainWindow", "Select File", None))
        self.checkBox_multi_map.setText(QCoreApplication.translate("MainWindow", "Multi-Map", None))
        self.lineEdit_output_splatted.setText(QCoreApplication.translate("MainWindow", "./workspace/splat", None))
        self.label_input_source.setText(QCoreApplication.translate("MainWindow", "Input Source Clips:", None))
        self.label_output_splatted.setText(QCoreApplication.translate("MainWindow", "Output Splatted:", None))
        self.pushButton_select_source.setText(QCoreApplication.translate("MainWindow", "Select File", None))
        self.pushButton_browse_source.setText(QCoreApplication.translate("MainWindow", "Browse Folder", None))
        self.lineEdit_input_depth.setText(QCoreApplication.translate("MainWindow", "./workspace/depth", None))
        self.pushButton_browse_sidecar.setText(QCoreApplication.translate("MainWindow", "Browse Folder", None))
        self.lineEdit_inout_sidecar.setText(QCoreApplication.translate("MainWindow", "./workspace/sidecar", None))
        self.label_output_sidecar.setText(QCoreApplication.translate("MainWindow", "I/O Sidecar:", None))
        self.label_frame_info.setText(QCoreApplication.translate("MainWindow", "Frame: 0 / 0", None))
        self.label_preview_source.setText(QCoreApplication.translate("MainWindow", "Preview Source:", None))
        self.comboBox_preview_source.setItemText(0, QCoreApplication.translate("MainWindow", "Splat Result", None))

        self.pushButton_load_refresh.setText(QCoreApplication.translate("MainWindow", "Load/Refresh List", None))
        # if QT_CONFIG(shortcut)
        self.pushButton_load_refresh.setShortcut(QCoreApplication.translate("MainWindow", "Return", None))
        # endif // QT_CONFIG(shortcut)
        self.pushButton_prev.setText(QCoreApplication.translate("MainWindow", "< Prev", None))
        # if QT_CONFIG(shortcut)
        self.pushButton_prev.setShortcut(QCoreApplication.translate("MainWindow", "Ctrl+Left", None))
        # endif // QT_CONFIG(shortcut)
        self.pushButton_next.setText(QCoreApplication.translate("MainWindow", "Next >", None))
        # if QT_CONFIG(shortcut)
        self.pushButton_next.setShortcut(QCoreApplication.translate("MainWindow", "Ctrl+Right", None))
        # endif // QT_CONFIG(shortcut)
        self.label_jump_to.setText(QCoreApplication.translate("MainWindow", "Jump to:", None))
        self.lineEdit_jump_to.setText(QCoreApplication.translate("MainWindow", "1", None))
        self.label_video_info.setText(QCoreApplication.translate("MainWindow", "Video: 0 / 0", None))
        self.pushButton_play.setText("")
        self.pushButton_fast_forward.setText("")
        self.pushButton_loop_toggle.setText("")
        self.label_depth_map.setText(QCoreApplication.translate("MainWindow", "Depth Map:", None))
        self.comboBox_map_select.setItemText(0, QCoreApplication.translate("MainWindow", "Default", None))
        self.comboBox_map_select.setItemText(1, "")

        self.label_preview_scale.setText(QCoreApplication.translate("MainWindow", "Preview Scale:", None))
        self.comboBox_preview_scale.setItemText(0, QCoreApplication.translate("MainWindow", "70%", None))

        self.groupBox_splatting_settings.setTitle(
            QCoreApplication.translate("MainWindow", "Splatting & Output Settings", None)
        )
        self.lineEdit_process_length.setText(QCoreApplication.translate("MainWindow", "-1", None))
        self.label_mesh_bias.setText(QCoreApplication.translate("MainWindow", "Mesh Bias:", None))
        self.label_border.setText(QCoreApplication.translate("MainWindow", "Border:", None))
        self.comboBox_border.setItemText(0, QCoreApplication.translate("MainWindow", "Auto Adv.", None))

        self.lineEdit_mesh_bias.setText(QCoreApplication.translate("MainWindow", "0.5", None))
        self.label_extrusion.setText(QCoreApplication.translate("MainWindow", "Extrusion:", None))
        self.label_process_length.setText(QCoreApplication.translate("MainWindow", "Process Length:", None))
        self.label_auto_convergence.setText(QCoreApplication.translate("MainWindow", "Auto-Converge:", None))
        self.lineEdit_mesh_extrusion.setText(QCoreApplication.translate("MainWindow", "0.5", None))
        self.comboBox_mask_type.setItemText(0, QCoreApplication.translate("MainWindow", "M2S", None))

        self.label_dolly.setText(QCoreApplication.translate("MainWindow", "Dolly:", None))
        self.label_density.setText(QCoreApplication.translate("MainWindow", "Density:", None))
        self.lineEdit_mesh_dolly.setText(QCoreApplication.translate("MainWindow", "0", None))
        self.label_mask.setText(QCoreApplication.translate("MainWindow", "Mask:", None))
        self.comboBox_auto_convergence.setItemText(0, QCoreApplication.translate("MainWindow", "Off", None))

        self.lineEdit_mesh_density.setText(QCoreApplication.translate("MainWindow", "0.5", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_output_settings),
            QCoreApplication.translate("MainWindow", "Output Settings", None),
        )
        self.groupBox_process_resolution.setTitle(QCoreApplication.translate("MainWindow", "Process Resolution", None))
        self.checkBox_enable_full_res.setText(QCoreApplication.translate("MainWindow", "Enable Full Res", None))
        self.label_batch_size_full.setText(QCoreApplication.translate("MainWindow", "Batch Size:", None))
        self.checkBox_enable_low_res.setText(QCoreApplication.translate("MainWindow", "Enable Low Res", None))
        self.label_batch_size_low.setText(QCoreApplication.translate("MainWindow", "Batch Size:", None))
        self.label_width.setText(QCoreApplication.translate("MainWindow", "Width:", None))
        self.label_height.setText(QCoreApplication.translate("MainWindow", "Height:", None))
        self.checkBox_dual_output.setText(QCoreApplication.translate("MainWindow", "Dual Output Only", None))
        self.checkBox_ffmpeg.setText(QCoreApplication.translate("MainWindow", "ffmpeg", None))
        self.lineEdit_high_batch.setText(QCoreApplication.translate("MainWindow", "7", None))
        self.lineEdit_low_batch.setText(QCoreApplication.translate("MainWindow", "13", None))
        self.lineEdit_low_height.setText(QCoreApplication.translate("MainWindow", "320", None))
        self.lineEdit_low_width.setText(QCoreApplication.translate("MainWindow", "640", None))
        self.groupBox_individual_outputs.setTitle(QCoreApplication.translate("MainWindow", "Individual Outputs", None))
        self.checkBox_splat_low.setText(QCoreApplication.translate("MainWindow", "Splat Low", None))
        self.checkBox_splat_hi.setText(QCoreApplication.translate("MainWindow", "Splat High", None))
        self.checkBox_anaglyph.setText(QCoreApplication.translate("MainWindow", "Analyph 3D", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", "Mask Low", None))
        self.checkBox_mask.setText(QCoreApplication.translate("MainWindow", "Mask High", None))
        self.checkBox_flowmap_x.setText(QCoreApplication.translate("MainWindow", "Flowmap X", None))
        self.checkBox_mesh_sbs.setText(QCoreApplication.translate("MainWindow", "Mesh SBS", None))
        self.checkBox_splat_sbs.setText(QCoreApplication.translate("MainWindow", "Splat SBS", None))
        self.checkBox_flowmap_full.setText(QCoreApplication.translate("MainWindow", "Flowmap Full", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_output_types),
            QCoreApplication.translate("MainWindow", "Output Types", None),
        )
        self.groupBox_depth_map_preprocessing.setTitle(
            QCoreApplication.translate("MainWindow", "Depth Map Pre-processing", None)
        )
        self.label_blur_left_value.setText(QCoreApplication.translate("MainWindow", "0", None))
        self.label_blur_y_value.setText(QCoreApplication.translate("MainWindow", "0", None))
        self.label_blur_y.setText(QCoreApplication.translate("MainWindow", "Blur Y:", None))
        self.label_dilate_left.setText(QCoreApplication.translate("MainWindow", "Dilate L:", None))
        self.label_dilate_y.setText(QCoreApplication.translate("MainWindow", "Dilate Y:", None))
        self.label_blur_left.setText(QCoreApplication.translate("MainWindow", "Blur L:", None))
        self.label_dilate_x.setText(QCoreApplication.translate("MainWindow", "Dilate X:", None))
        self.label_blur_x.setText(QCoreApplication.translate("MainWindow", "Blur X:", None))
        self.label_blue_x_value.setText(QCoreApplication.translate("MainWindow", "1", None))
        self.label_dilate_y_value.setText(QCoreApplication.translate("MainWindow", "3.5", None))
        self.lineEdit_blur_bias.setText(QCoreApplication.translate("MainWindow", "0.5", None))
        self.label_dilate_left_value.setText(QCoreApplication.translate("MainWindow", "0.0", None))
        self.label_dilate_x_value.setText(QCoreApplication.translate("MainWindow", "7", None))
        self.groupBox_stereo_projection.setTitle(QCoreApplication.translate("MainWindow", "Stereo Projection", None))
        self.label_gamma.setText(QCoreApplication.translate("MainWindow", "Gamma:", None))
        self.label_gamma_value.setText(QCoreApplication.translate("MainWindow", "1.0", None))
        self.label_disparity.setText(QCoreApplication.translate("MainWindow", "Disparity:", None))
        self.label_disparity_value.setText(QCoreApplication.translate("MainWindow", "35", None))
        self.label_convergence.setText(QCoreApplication.translate("MainWindow", "Convergence:", None))
        self.label_convergence_value.setText(QCoreApplication.translate("MainWindow", "1.0", None))
        self.label_border_width.setText(QCoreApplication.translate("MainWindow", "Border Width:", None))
        self.label_border_width_value.setText(QCoreApplication.translate("MainWindow", "0.0", None))
        self.label_bias.setText(QCoreApplication.translate("MainWindow", "Border Bias:", None))
        self.label_bias_value.setText(QCoreApplication.translate("MainWindow", "0.5", None))
        self.checkBox_normalization.setText(QCoreApplication.translate("MainWindow", "Global Normalization", None))
        self.checkBox_resume.setText(QCoreApplication.translate("MainWindow", "Resume", None))
        self.checkBox_cross_view.setText(QCoreApplication.translate("MainWindow", "Cross View", None))
        # if QT_CONFIG(shortcut)
        self.checkBox_cross_view.setShortcut(QCoreApplication.translate("MainWindow", "X", None))
        # endif // QT_CONFIG(shortcut)
        self.groupBox_current_processing_info.setTitle(
            QCoreApplication.translate("MainWindow", "Current Processing Information", None)
        )
        self.label_info_gamma_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_info_map_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_info_frames_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_info_disparity_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_gamma_info.setText(QCoreApplication.translate("MainWindow", "Gamma:", None))
        self.label_convergence_info.setText(QCoreApplication.translate("MainWindow", "Converge:", None))
        self.label_filename_info.setText(QCoreApplication.translate("MainWindow", "Filename:", None))
        self.label_resolution_info.setText(QCoreApplication.translate("MainWindow", "Resolution:", None))
        self.label_task_info.setText(QCoreApplication.translate("MainWindow", "Task:", None))
        self.label_frames_info.setText(QCoreApplication.translate("MainWindow", "Frames:", None))
        self.label_info_filename_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_info_convergence_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_info_resolution_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.label_disparity_info.setText(QCoreApplication.translate("MainWindow", "Disparity:", None))
        self.label_map_info.setText(QCoreApplication.translate("MainWindow", "Map:", None))
        self.label_info_task_value.setText(QCoreApplication.translate("MainWindow", "N/A", None))
        self.groupBox_dev_tools.setTitle(QCoreApplication.translate("MainWindow", "Dev Tools", None))
        self.checkBox_crosshair.setText(QCoreApplication.translate("MainWindow", "Crosshair", None))
        self.checkBox_dp.setText(QCoreApplication.translate("MainWindow", "D/P", None))
        self.checkBox_multi.setText(QCoreApplication.translate("MainWindow", "Multi", None))
        self.checkBox_splat_test.setText(QCoreApplication.translate("MainWindow", "Splat Test", None))
        self.checkBox_white.setText(QCoreApplication.translate("MainWindow", "White", None))
        self.checkBox_true_max.setText(QCoreApplication.translate("MainWindow", "True Max", None))
        self.checkBox_map_test.setText(QCoreApplication.translate("MainWindow", "Map Test", None))
        self.checkBox_skip_low_res.setText(QCoreApplication.translate("MainWindow", "Skip Low-Res Pre-proc", None))
        self.progressBar.setFormat(QCoreApplication.translate("MainWindow", "Progress:", None))
        self.label_status.setText(QCoreApplication.translate("MainWindow", "Ready", None))
        self.pushButton_single.setText(QCoreApplication.translate("MainWindow", "SINGLE", None))
        self.pushButton_start.setText(QCoreApplication.translate("MainWindow", "START", None))
        self.label_from.setText(QCoreApplication.translate("MainWindow", "From:", None))
        self.label_to.setText(QCoreApplication.translate("MainWindow", "To:", None))
        self.pushButton_stop.setText(QCoreApplication.translate("MainWindow", "STOP", None))
        self.pushButton_preview_auto_converge.setText(
            QCoreApplication.translate("MainWindow", "Preview Auto-Converge", None)
        )
        self.pushButton_auto_pass.setText(QCoreApplication.translate("MainWindow", "AUTO-PASS", None))
        self.pushButton_update_sidecar.setText(QCoreApplication.translate("MainWindow", "Update Sidecar", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", "File", None))
        self.menuOptions.setTitle(QCoreApplication.translate("MainWindow", "Options", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", "Help", None))

    # retranslateUi
