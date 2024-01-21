import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap


class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("打开图片")
        self.label = QLabel(self)
        self.button = QPushButton("衣服", self)
        self.button.clicked.connect(self.open_image_clothes)

        self.button1 = QPushButton("模特", self)
        self.button1.clicked.connect(self.open_image_model)


        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def open_image_clothes(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        # if filepath:
        # filepath = 'study/1.jpg'
        pixmap = QPixmap(filepath)
        self.label.setPixmap(pixmap.scaled(200, 180, aspectRatioMode=False))

    def open_image_model(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        # if filepath:
        # filepath = 'study/1.jpg'
        pixmap = QPixmap(filepath)
        self.label.setPixmap(pixmap.scaled(200, 180, aspectRatioMode=False))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec_())
