import sys
from virtual_try_on import Ui_MainWindow#调用ui模块
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QLabel,QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
class MyMainWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.setupUi(self)

        # 连接按钮的点击事件到自定义方法
        self.pushButton.clicked.connect(self.open_image)
        # dialog = QFileDialog()  # 创建文件对话框
        # filepath, _ = dialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        # filepath = 'study/1.jpg'
        # if filepath:
        #     pixmap = QPixmap(filepath)
        #     self.label.setPixmap(pixmap.scaled(800, 600))
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pushButton)
        self.setLayout(layout)

    def open_image(self):
        dialog = QFileDialog()  # 创建文件对话框
        filepath, _ = dialog.getOpenFileName(self, "衣服", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        # filepath = 'study/1.jpg'
        if filepath:
            pixmap = QPixmap(filepath)
            self.label.setPixmap(pixmap.scaled(800, 600))


if __name__ == "__main__":

    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())