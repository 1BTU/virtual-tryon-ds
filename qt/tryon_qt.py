'''打开电脑图片并显示'''
import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def pf():
    print('123456')


class Picture(QMainWindow):
    def __init__(self):
        super(Picture, self).__init__()

        self.resize(1000, 800)
        self.setWindowTitle("label显示图片")
        #衣服
        self.label = QLabel(self)
        # self.label.setText("   显示图片")
        self.label.setFixedSize(int(1.5*192), int(1.5*256))
        self.label.move(1, 160)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn = QPushButton(self)
        btn.setText("衣服")
        btn.move(10, 30)
        btn.clicked.connect(self.openimage_cloth)

        #模特
        self.label1 = QLabel(self)
        # self.label.setText("   显示图片")
        self.label1.setFixedSize(int(1.5*192), int(1.5*256))
        self.label1.move(int(1.5*192)+20, 160)

        self.label1.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn1 = QPushButton(self)
        btn1.setText("模特")
        btn1.move(10+int(1.5*192), 30)
        btn1.clicked.connect(self.openimage_model)
        # #试衣
        self.label2 = QLabel(self)
        # self.label.setText("   显示图片")
        self.label2.setFixedSize(int(1.5*192), int(1.5*256))
        self.label2.move(2*int(1.5*192)+40, 160)

        self.label2.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )

        btn2 = QPushButton(self)
        btn2.setText("试衣")
        btn2.move(10+2*int(1.5*192), 30)
        btn2.clicked.connect(self.return_image)

    def openimage_cloth(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "衣服", "", "*.jpg;;*.png;;All Files(*)")  # 在电脑上加代码
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        # pf()
    def openimage_model(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "模特", "", "*.jpg;;*.png;;All Files(*)")  # 在电脑上加代码
        jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
        self.label1.setPixmap(jpg)

    def show_image_in_label(self, image):
        height, width, channel = image.shape
        bytes_per_line = channel * width
        qimage = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.label2.width(), self.label2.height())
        self.label2.setPixmap(scaled_pixmap)

    def return_image(self):
        # 示例用法
        image_data_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # 替换为您的RGB图像数据
        self.show_image_in_label(image_data_rgb)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = Picture()
    my.show()
    sys.exit(app.exec_())