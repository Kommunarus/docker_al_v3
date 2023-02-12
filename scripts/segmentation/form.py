from PyQt6 import QtCore, QtGui, QtWidgets
import sys
import os
from scripts.segmentation.eval import inference


class UiForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.list_id = sorted(os.listdir(path_to_dir))

        self.type = QtWidgets.QCheckBox('box/segm')
        self.graphicsView = QtWidgets.QLabel()
        self.graphicsView_2 = QtWidgets.QLabel()
        self.pushButton = QtWidgets.QPushButton('O')
        self.pushButton_2 = QtWidgets.QPushButton('<')
        self.pushButton_3 = QtWidgets.QPushButton('>')
        # self.text_metric = QtWidgets.QTextEdit()
        # self.text_metric_2 = QtWidgets.QTextEdit()
        self.table = QtWidgets.QTableView()
        self.sti = QtGui.QStandardItemModel()

        self.list = QtWidgets.QComboBox()
        self.addId()

        self.label = QtWidgets.QLabel('Id: ')
        self.label_metric = QtWidgets.QLabel('maP / maR')

        self.pushButton.clicked.connect(self.curr)
        self.pushButton_2.clicked.connect(self.prev)
        self.pushButton_3.clicked.connect(self.next)


        self.vbox = QtWidgets.QVBoxLayout()

        self.hbox1 = QtWidgets.QHBoxLayout()
        self.hbox1.addWidget(self.label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.hbox1.addWidget(self.list)
        self.hbox1.addWidget(self.type)
        self.hbox1.addWidget(self.pushButton)
        self.hbox1.addWidget(self.pushButton_2)
        self.hbox1.addWidget(self.pushButton_3)


        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.addWidget(self.graphicsView, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.hbox2.addWidget(self.graphicsView_2, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        # self.hbox3 = QtWidgets.QHBoxLayout()
        # self.hbox3.addWidget(self.text_metric, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        # self.hbox3.addWidget(self.text_metric_2, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        # self.vbox.addLayout(self.hbox3)

        self.vbox.addWidget(self.label_metric)
        self.vbox.addWidget(self.table)

        self.setLayout(self.vbox)

    def on_show(self, id):
        namemodel = './rcnn_s.pth'
        namefile = './img.jpg'
        mape, _, _ = inference([id, ], self.type.checkState().value, False, namemodel, namefile, True)
        self.graphicsView_2.setPixmap(QtGui.QPixmap('./img.jpg'))
        self.graphicsView_2.show()

        self.graphicsView.setPixmap(QtGui.QPixmap('./img_tr.jpg'))
        self.graphicsView.show()

        self.sti.clear()
        for k, v in mape.items():
            if k in ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'mar_small', 'mar_medium']:
                item1 = QtGui.QStandardItem(k)
                item2 = QtGui.QStandardItem(str(round(v.item(), 3)))
                self.sti.appendRow([item1, item2])
        self.sti.setHorizontalHeaderLabels(['param', 'value'])
        self.table.setModel(self.sti)
        self.table.setColumnWidth(0, 250)
        self.table.setColumnWidth(1, 100)




    def curr(self):
        id = self.list.currentText()
        self.on_show(id)

    def prev(self):
        ind = self.list.currentIndex()
        id_prev = ind - 1
        if id_prev < 0:
            id_prev = len(self.list_id) - 1
        self.list.setCurrentIndex(id_prev)
        self.on_show(self.list.currentText())

    def next(self):
        ind = self.list.currentIndex()
        id_next = ind + 1
        if id_next == len(self.list_id):
            id_next = 0
        self.list.setCurrentIndex(id_next)
        self.on_show(self.list.currentText())

    def addId(self):
        for roe in self.list_id:
            self.list.addItem(roe)


if __name__ == '__main__':
    path_to_dir = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
    app = QtWidgets.QApplication(sys.argv)
    form = UiForm()
    form.setFixedSize(850, 720)
    form.show()
    sys.exit(app.exec())
