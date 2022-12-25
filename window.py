from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image, ImageOps


class Ui_Braille_Image_to_text(object):
    def calc(self):
        path = self.textEdit.toPlainText()

        numero_lettre = 0
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t","u", "v", "w", "x", "y", "z"]
        model = load_model('braille.h5')

        img = Image.open(path)
        img = img.convert('L')
        img.save(path)

        img = image.load_img(path)
        img = image.img_to_array(img)
        img = cv2.resize(img, (28, 28))

        x = np.expand_dims(img, axis=0)
        a = model.predict(x)
        array_value = []
        for j in range(len(a[0])):
            array_value.append(a[0][j])

        a = np.argmax(model.predict(x), axis=1)
        numero_lettre = a[0]
        lettre = str(alphabet[numero_lettre])
        acc = str(array_value[numero_lettre] * 100)

        self.textEdit_2.setText("la lettre est : "+lettre + " avec " + acc[0:6]+" de précision")

    def setupUi(self, Image_To_Text):
        Image_To_Text.setObjectName("Image_To_Text")
        Image_To_Text.resize(768, 615)
        self.pushButton = QtWidgets.QPushButton(Image_To_Text)
        self.pushButton.setGeometry(QtCore.QRect(270, 450, 201, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")

        self.pushButton.clicked.connect(self.calc)

        self.textEdit = QtWidgets.QTextEdit(Image_To_Text)
        self.textEdit.setGeometry(QtCore.QRect(230, 180, 461, 91))

        self.textEdit.setObjectName("textEdit")

        self.label = QtWidgets.QLabel(Image_To_Text)
        self.label.setGeometry(QtCore.QRect(70, 190, 141, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Image_To_Text)
        self.label_2.setGeometry(QtCore.QRect(70, 310, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textEdit_2 = QtWidgets.QTextEdit(Image_To_Text)
        self.textEdit_2.setGeometry(QtCore.QRect(230, 300, 461, 91))

        self.textEdit_2.setObjectName("textEdit_2")

        self.retranslateUi(Image_To_Text)
        QtCore.QMetaObject.connectSlotsByName(Image_To_Text)

    def retranslateUi(self, Image_To_Text):
        _translate = QtCore.QCoreApplication.translate
        Image_To_Text.setWindowTitle(_translate("Image_To_Text", "Braille Convertisseur"))
        self.pushButton.setText(_translate("Image_To_Text", "Convertir"))

        self.label.setText(_translate("Image_To_Text", "Source Image"))
        self.label_2.setText(_translate("Image_To_Text", "Résultat"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Image_To_Text = QtWidgets.QWidget()
    ui = Ui_Braille_Image_to_text()
    ui.setupUi(Image_To_Text)
    Image_To_Text.show()
    sys.exit(app.exec_())