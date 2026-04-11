# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'encode.ui'
##
## Created by: Qt User Interface Compiler version 6.11.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(399, 347)
        self.gridLayout_3 = QGridLayout(Dialog)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.StandardButton.Cancel|QDialogButtonBox.StandardButton.Ok)

        self.gridLayout_3.addWidget(self.buttonBox, 2, 1, 1, 1)

        self.groupBox_output_format = QGroupBox(Dialog)
        self.groupBox_output_format.setObjectName(u"groupBox_output_format")
        self.gridLayout = QGridLayout(self.groupBox_output_format)
        self.gridLayout.setObjectName(u"gridLayout")
        self.comboBox_container = QComboBox(self.groupBox_output_format)
        self.comboBox_container.setObjectName(u"comboBox_container")

        self.gridLayout.addWidget(self.comboBox_container, 1, 1, 1, 1)

        self.lineEdit_full_res = QLineEdit(self.groupBox_output_format)
        self.lineEdit_full_res.setObjectName(u"lineEdit_full_res")

        self.gridLayout.addWidget(self.lineEdit_full_res, 8, 1, 1, 1)

        self.comboBox_codec = QComboBox(self.groupBox_output_format)
        self.comboBox_codec.setObjectName(u"comboBox_codec")

        self.gridLayout.addWidget(self.comboBox_codec, 0, 1, 1, 1)

        self.comboBox_cpu_tune = QComboBox(self.groupBox_output_format)
        self.comboBox_cpu_tune.setObjectName(u"comboBox_cpu_tune")

        self.gridLayout.addWidget(self.comboBox_cpu_tune, 4, 1, 1, 1)

        self.label_container = QLabel(self.groupBox_output_format)
        self.label_container.setObjectName(u"label_container")
        self.label_container.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_container, 1, 0, 1, 1)

        self.label_crf_high = QLabel(self.groupBox_output_format)
        self.label_crf_high.setObjectName(u"label_crf_high")
        self.label_crf_high.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_crf_high, 8, 0, 1, 1)

        self.label_cpu_tune = QLabel(self.groupBox_output_format)
        self.label_cpu_tune.setObjectName(u"label_cpu_tune")
        self.label_cpu_tune.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_cpu_tune, 4, 0, 1, 1)

        self.comboBox_encoder = QComboBox(self.groupBox_output_format)
        self.comboBox_encoder.setObjectName(u"comboBox_encoder")

        self.gridLayout.addWidget(self.comboBox_encoder, 2, 1, 1, 1)

        self.lineEdit_low_res = QLineEdit(self.groupBox_output_format)
        self.lineEdit_low_res.setObjectName(u"lineEdit_low_res")

        self.gridLayout.addWidget(self.lineEdit_low_res, 7, 1, 1, 1)

        self.label_crf_low = QLabel(self.groupBox_output_format)
        self.label_crf_low.setObjectName(u"label_crf_low")
        self.label_crf_low.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_crf_low, 7, 0, 1, 1)

        self.comboBox_quality = QComboBox(self.groupBox_output_format)
        self.comboBox_quality.setObjectName(u"comboBox_quality")

        self.gridLayout.addWidget(self.comboBox_quality, 3, 1, 1, 1)

        self.label_color_tag = QLabel(self.groupBox_output_format)
        self.label_color_tag.setObjectName(u"label_color_tag")
        self.label_color_tag.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_color_tag, 5, 0, 1, 1)

        self.label_encoder = QLabel(self.groupBox_output_format)
        self.label_encoder.setObjectName(u"label_encoder")
        self.label_encoder.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_encoder, 2, 0, 1, 1)

        self.comboBox_color_tag = QComboBox(self.groupBox_output_format)
        self.comboBox_color_tag.setObjectName(u"comboBox_color_tag")

        self.gridLayout.addWidget(self.comboBox_color_tag, 5, 1, 1, 1)

        self.label_quality = QLabel(self.groupBox_output_format)
        self.label_quality.setObjectName(u"label_quality")
        self.label_quality.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_quality, 3, 0, 1, 1)

        self.label_codec = QLabel(self.groupBox_output_format)
        self.label_codec.setObjectName(u"label_codec")
        self.label_codec.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.label_codec, 0, 0, 1, 1)

        self.line = QFrame(self.groupBox_output_format)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line, 6, 0, 1, 2)


        self.gridLayout_3.addWidget(self.groupBox_output_format, 0, 0, 2, 1)

        self.groupBox_nvenco_options = QGroupBox(Dialog)
        self.groupBox_nvenco_options.setObjectName(u"groupBox_nvenco_options")
        self.verticalLayout = QVBoxLayout(self.groupBox_nvenco_options)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.checkBox_lookahead = QCheckBox(self.groupBox_nvenco_options)
        self.checkBox_lookahead.setObjectName(u"checkBox_lookahead")
        self.checkBox_lookahead.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.verticalLayout.addWidget(self.checkBox_lookahead)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.comboBox_look_ahead_frames = QComboBox(self.groupBox_nvenco_options)
        self.comboBox_look_ahead_frames.setObjectName(u"comboBox_look_ahead_frames")
        self.comboBox_look_ahead_frames.setEnabled(False)
        self.comboBox_look_ahead_frames.setMinimumSize(QSize(60, 0))
        self.comboBox_look_ahead_frames.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout_2.addWidget(self.comboBox_look_ahead_frames)

        self.label_lookahead_frames = QLabel(self.groupBox_nvenco_options)
        self.label_lookahead_frames.setObjectName(u"label_lookahead_frames")
        self.label_lookahead_frames.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout_2.addWidget(self.label_lookahead_frames)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.checkBox_spatial_aq = QCheckBox(self.groupBox_nvenco_options)
        self.checkBox_spatial_aq.setObjectName(u"checkBox_spatial_aq")
        self.checkBox_spatial_aq.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.verticalLayout.addWidget(self.checkBox_spatial_aq)

        self.checkBox_temporal_aq = QCheckBox(self.groupBox_nvenco_options)
        self.checkBox_temporal_aq.setObjectName(u"checkBox_temporal_aq")
        self.checkBox_temporal_aq.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.verticalLayout.addWidget(self.checkBox_temporal_aq)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.comboBox_strength_aq = QComboBox(self.groupBox_nvenco_options)
        self.comboBox_strength_aq.setObjectName(u"comboBox_strength_aq")
        self.comboBox_strength_aq.setMinimumSize(QSize(60, 0))
        self.comboBox_strength_aq.setMaximumSize(QSize(60, 16777215))

        self.horizontalLayout.addWidget(self.comboBox_strength_aq)

        self.label_strength_aq = QLabel(self.groupBox_nvenco_options)
        self.label_strength_aq.setObjectName(u"label_strength_aq")
        self.label_strength_aq.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout.addWidget(self.label_strength_aq)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.gridLayout_3.addWidget(self.groupBox_nvenco_options, 0, 1, 1, 1)

        QWidget.setTabOrder(self.comboBox_codec, self.comboBox_container)
        QWidget.setTabOrder(self.comboBox_container, self.comboBox_encoder)
        QWidget.setTabOrder(self.comboBox_encoder, self.comboBox_quality)
        QWidget.setTabOrder(self.comboBox_quality, self.comboBox_cpu_tune)
        QWidget.setTabOrder(self.comboBox_cpu_tune, self.comboBox_color_tag)
        QWidget.setTabOrder(self.comboBox_color_tag, self.lineEdit_low_res)
        QWidget.setTabOrder(self.lineEdit_low_res, self.lineEdit_full_res)
        QWidget.setTabOrder(self.lineEdit_full_res, self.checkBox_lookahead)
        QWidget.setTabOrder(self.checkBox_lookahead, self.comboBox_look_ahead_frames)
        QWidget.setTabOrder(self.comboBox_look_ahead_frames, self.checkBox_spatial_aq)
        QWidget.setTabOrder(self.checkBox_spatial_aq, self.checkBox_temporal_aq)
        QWidget.setTabOrder(self.checkBox_temporal_aq, self.comboBox_strength_aq)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.groupBox_output_format.setTitle(QCoreApplication.translate("Dialog", u"Output Format", None))
        self.label_container.setText(QCoreApplication.translate("Dialog", u"Container:", None))
        self.label_crf_high.setText(QCoreApplication.translate("Dialog", u"Full Res CRF:", None))
        self.label_cpu_tune.setText(QCoreApplication.translate("Dialog", u"CPU Tune:", None))
        self.label_crf_low.setText(QCoreApplication.translate("Dialog", u"Low Res CRF:", None))
        self.label_color_tag.setText(QCoreApplication.translate("Dialog", u"Color Tages", None))
        self.label_encoder.setText(QCoreApplication.translate("Dialog", u"Encoder:", None))
        self.label_quality.setText(QCoreApplication.translate("Dialog", u"Quality Preset:", None))
        self.label_codec.setText(QCoreApplication.translate("Dialog", u"Codec:", None))
        self.groupBox_nvenco_options.setTitle(QCoreApplication.translate("Dialog", u"NVENC ONLY Options", None))
        self.checkBox_lookahead.setStyleSheet(QCoreApplication.translate("Dialog", u"0", None))
        self.checkBox_lookahead.setText(QCoreApplication.translate("Dialog", u"Enable Lookahead", None))
        self.label_lookahead_frames.setText(QCoreApplication.translate("Dialog", u"Lookahead Frames", None))
        self.checkBox_spatial_aq.setStyleSheet(QCoreApplication.translate("Dialog", u"0", None))
        self.checkBox_spatial_aq.setText(QCoreApplication.translate("Dialog", u"Spatial AQ", None))
        self.checkBox_temporal_aq.setStyleSheet(QCoreApplication.translate("Dialog", u"0", None))
        self.checkBox_temporal_aq.setText(QCoreApplication.translate("Dialog", u"Temporal AQ", None))
        self.label_strength_aq.setText(QCoreApplication.translate("Dialog", u"Strength AQ", None))
    # retranslateUi

