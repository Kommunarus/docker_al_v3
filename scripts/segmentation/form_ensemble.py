from PyQt6 import QtCore, QtGui, QtWidgets
import sys
import os
from scripts.segmentation.eval import inference
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks


class UiForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.list_id = sorted(os.listdir(path_to_dir))

        self.type = QtWidgets.QCheckBox('box/segm')
        self.graphicsView = QtWidgets.QLabel()
        self.graphicsView_mod1 = QtWidgets.QLabel()
        self.graphicsView_mod2 = QtWidgets.QLabel()
        self.graphicsView_mod3 = QtWidgets.QLabel()

        self.graphicsView_i = QtWidgets.QLabel()
        self.graphicsView_u = QtWidgets.QLabel()

        self.pushButton = QtWidgets.QPushButton('O')
        self.pushButton_2 = QtWidgets.QPushButton('<')
        self.pushButton_3 = QtWidgets.QPushButton('>')
        self.pushButton_4 = QtWidgets.QPushButton('S')
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
        self.pushButton_4.clicked.connect(self.sort)


        self.vbox = QtWidgets.QVBoxLayout()

        self.hbox1 = QtWidgets.QHBoxLayout()
        self.hbox1.addWidget(self.label, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.hbox1.addWidget(self.list)
        self.hbox1.addWidget(self.type)
        self.hbox1.addWidget(self.pushButton)
        self.hbox1.addWidget(self.pushButton_2)
        self.hbox1.addWidget(self.pushButton_3)
        self.hbox1.addWidget(self.pushButton_4)


        self.hbox2 = QtWidgets.QHBoxLayout()
        self.hbox2.addWidget(self.graphicsView, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.hbox2.addWidget(self.graphicsView_i, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.hbox2.addWidget(self.graphicsView_u, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        self.hbox3 = QtWidgets.QHBoxLayout()
        self.hbox3.addWidget(self.graphicsView_mod1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.hbox3.addWidget(self.graphicsView_mod2, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.hbox3.addWidget(self.graphicsView_mod3, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        # self.hbox4 = QtWidgets.QHBoxLayout()
        # self.hbox4.addWidget(self.graphicsView_i, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        # self.hbox4.addWidget(self.graphicsView_u, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addLayout(self.hbox3)
        # self.vbox.addLayout(self.hbox4)

        self.vbox.addWidget(self.label_metric)
        # self.vbox.addWidget(self.table)

        self.setLayout(self.vbox)

    def on_show(self, id):

        a, b = self.findIoU(id)

        a1 = a.int().sum().item()
        b1 = b.int().sum().item()
        self.label_metric.setText(str(1 - a1 / b1))

        fig, ax = plt.subplots()
        img = 255*torch.ones((3, 224, 224)).to(torch.uint8)
        img = draw_segmentation_masks(img, a, alpha=0.8)
        ax.imshow(torch.movedim(img, 0, 2))
        ax.set_axis_off()
        plt.savefig('img_i.jpg', bbox_inches='tight')

        fig, ax = plt.subplots()
        img = 255*torch.ones((3, 224, 224)).to(torch.uint8)
        img = draw_segmentation_masks(img, b, alpha=0.8)
        ax.imshow(torch.movedim(img, 0, 2))
        ax.set_axis_off()
        plt.savefig('img_o.jpg', bbox_inches='tight')

        self.graphicsView.setPixmap(QtGui.QPixmap('./img_tr.jpg'))
        self.graphicsView.show()

        self.graphicsView_mod1.setPixmap(QtGui.QPixmap('./img_m1.jpg'))
        self.graphicsView_mod1.show()

        self.graphicsView_mod2.setPixmap(QtGui.QPixmap('./img_m2.jpg'))
        self.graphicsView_mod2.show()

        self.graphicsView_mod3.setPixmap(QtGui.QPixmap('./img_m3.jpg'))
        self.graphicsView_mod3.show()

        self.graphicsView_i.setPixmap(QtGui.QPixmap('./img_i.jpg'))
        self.graphicsView_i.show()

        self.graphicsView_u.setPixmap(QtGui.QPixmap('./img_o.jpg'))
        self.graphicsView_u.show()


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

    def sort(self):
        ll = {}
        for roe in self.list_id:
            try:
                a, b = self.findIoU(roe)
                a1 = a.int().sum().item()
                b1 = b.int().sum().item()

                k = 1 - a1/b1
                ll[roe] = k
            except:
                print('err', roe)

        rr = sorted(ll.items(), key=lambda x: x[1])
        print(rr[:10])
        print(rr[-10:])

    def findIoU(self, id):
        namemodel = './copy model/model 1.pth'
        namefile = './img_m1.jpg'
        _, pred_map, target_map = inference([id, ], self.type.checkState().value, False, namemodel, namefile, True)
        # t1 = torch.movedim(pred_map[0]['scores'] * torch.movedim(pred_map[0]['masks'].int(), 0, 2), 2, 0)
        t1 = (pred_map[0]['masks'] > 0.5).int()
        total_mask1 = torch.max(t1, 0)[0].cpu().numpy()

        namemodel = './copy model/model 2.pth'
        namefile = './img_m2.jpg'
        _, pred_map, target_map = inference([id, ], self.type.checkState().value, False, namemodel, namefile, False)
        # t1 = torch.movedim(pred_map[0]['scores'] * torch.movedim(pred_map[0]['masks'].int(), 0, 2), 2, 0)
        t1 = (pred_map[0]['masks'] > 0.5).int()
        total_mask2 = torch.max(t1, 0)[0].cpu().numpy()

        namemodel = './copy model/model 3.pth'
        namefile = './img_m3.jpg'
        _, pred_map, target_map = inference([id, ], self.type.checkState().value, False, namemodel, namefile, False)
        # t1 = torch.movedim(pred_map[0]['scores'] * torch.movedim(pred_map[0]['masks'].int(), 0, 2), 2, 0)
        t1 = (pred_map[0]['masks'] > 0.5).int()
        total_mask3 = torch.max(t1, 0)[0].cpu().numpy()

        a = torch.tensor(fI([total_mask1, total_mask2, total_mask3]))
        b = torch.tensor(OI([total_mask1, total_mask2, total_mask3]))

        return a, b

def fI(masks):
    s1 = np.prod(np.array(masks), 0)
    s2 = np.log(s1 + 0.01)
    s3 = s2 / len(masks)
    x = np.exp(s3)
    out = (x > 0.5)
    return out

def OI(masks):
    s1 = np.max(np.array(masks), 0)
    out = (s1 > 0.5)
    return out


if __name__ == '__main__':
    path_to_dir = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
    app = QtWidgets.QApplication(sys.argv)
    form = UiForm()
    # form.setFixedSize(850, 720)
    form.show()
    sys.exit(app.exec())
