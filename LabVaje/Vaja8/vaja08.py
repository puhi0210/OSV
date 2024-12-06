# -*- coding: utf-8 -*-

import os
import sys

from matplotlib.backends.qt_compat import QtCore
from PyQt5 import QtGui, QtWidgets

if int(QtCore.qVersion()[0]) == 5:
    from matplotlib.backends.backend_qt5agg import FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )

import matplotlib.pyplot as pp
import numpy as np
from matplotlib.figure import Figure


def warning(text):

    message = QtWidgets.QMessageBox()
    message.setIcon(QtWidgets.QMessageBox.Warning)
    message.setText(text)
    message.setInformativeText("Vnesite potrebne podatke.")
    message.addButton(QtWidgets.QMessageBox.Ok)
    message.exec()


numpytypes = {
    "8 bit": np.uint8,
    "16 bit": np.uint16,
}


class ui_RightPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):

        super().__init__()

        self.setFixedWidth(300)

        # nalaganje slike ##########

        group_loading_images = QtWidgets.QGroupBox("Nalaganje slike", self)
        hbox_x = QtWidgets.QHBoxLayout()
        text_x = QtWidgets.QLabel("Širina:", self)
        self.input_x = QtWidgets.QLineEdit(self)
        self.input_x.setValidator(QtGui.QIntValidator())
        self.input_x.setPlaceholderText("Vnesi število")
        hbox_x.addWidget(text_x)
        hbox_x.addWidget(self.input_x)

        hbox_y = QtWidgets.QHBoxLayout()
        text_y = QtWidgets.QLabel("Višina:", self)
        self.input_y = QtWidgets.QLineEdit(self)
        self.input_y.setValidator(QtGui.QIntValidator())
        self.input_y.setPlaceholderText("Vnesi število")
        hbox_y.addWidget(text_y)
        hbox_y.addWidget(self.input_y)

        hbox_bits = QtWidgets.QHBoxLayout()
        text_bits = QtWidgets.QLabel("Št. bitov:", self)
        text_bits.setFixedWidth(50)
        self.input_bits = QtWidgets.QComboBox(self)
        self.input_bits.addItem("8 bit")
        self.input_bits.addItem("16 bit")
        hbox_bits.addWidget(text_bits)
        hbox_bits.addWidget(self.input_bits)
        hbox_bits.setSpacing(0)

        self.load_button = QtWidgets.QPushButton("Naloži sliko...", self)
        self.clear_button = QtWidgets.QPushButton("Počisti sliko in podatke", self)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox_x)
        vbox.addLayout(hbox_y)
        vbox.addLayout(hbox_bits)
        vbox.addWidget(self.load_button)
        vbox.addWidget(self.clear_button)
        group_loading_images.setLayout(vbox)

        ##########

        # iskanje robov ##########

        group_edges = QtWidgets.QGroupBox("Iskanje robov", self)
        hbox_std = QtWidgets.QHBoxLayout()
        text_std = QtWidgets.QLabel("Standardni odklon:", self)
        self.input_std = QtWidgets.QLineEdit(self)
        self.input_std.setValidator(QtGui.QDoubleValidator())
        self.input_std.setPlaceholderText("Vnesi število")
        hbox_std.addWidget(text_std)
        hbox_std.addSpacing(1)
        hbox_std.addWidget(self.input_std)

        hbox_sprag = QtWidgets.QHBoxLayout()
        hbox_zprag = QtWidgets.QHBoxLayout()
        text_sprag = QtWidgets.QLabel("Spodnji prag:", self)
        text_zprag = QtWidgets.QLabel("Zgornji prag:", self)
        self.input_sprag = QtWidgets.QLineEdit(self)
        self.input_sprag.setValidator(QtGui.QDoubleValidator())
        self.input_zprag = QtWidgets.QLineEdit(self)
        self.input_zprag.setValidator(QtGui.QDoubleValidator())
        self.input_sprag.setPlaceholderText("Vnesi število od 0 do 1")
        self.input_zprag.setPlaceholderText("Vnesi število od 0 do 1")
        hbox_sprag.addWidget(text_sprag)
        hbox_sprag.addWidget(self.input_sprag)
        hbox_zprag.addWidget(text_zprag)
        hbox_zprag.addWidget(self.input_zprag)

        self.edges_button = QtWidgets.QPushButton("Poišči robove", self)

        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addLayout(hbox_std)
        vbox2.addLayout(hbox_sprag)
        vbox2.addLayout(hbox_zprag)
        vbox2.addWidget(self.edges_button)
        group_edges.setLayout(vbox2)

        ##########

        # houghova preslikava  ##########

        group_hough = QtWidgets.QGroupBox("Houghova preslikava", self)

        hbox_r = QtWidgets.QHBoxLayout()
        text_r = QtWidgets.QLabel("Korak parametra r [px]", self)
        self.input_r = QtWidgets.QLineEdit(self)
        self.input_r.setValidator(QtGui.QDoubleValidator())
        self.input_r.setPlaceholderText("Vnesi število")
        hbox_r.addWidget(text_r)
        hbox_r.addWidget(self.input_r)

        hbox_fi = QtWidgets.QHBoxLayout()
        text_fi = QtWidgets.QLabel("Korak parametra \u03C6 [deg]", self)
        self.input_fi = QtWidgets.QLineEdit(self)
        self.input_fi.setValidator(QtGui.QDoubleValidator())
        self.input_fi.setPlaceholderText("Vnesi število")
        hbox_fi.addWidget(text_fi)
        hbox_fi.addWidget(self.input_fi)

        self.hough_button = QtWidgets.QPushButton("Izvedi Houghovo preslikavo", self)

        vbox3 = QtWidgets.QVBoxLayout()
        vbox3.addLayout(hbox_r)
        vbox3.addLayout(hbox_fi)
        vbox3.addWidget(self.hough_button)
        group_hough.setLayout(vbox3)

        ##########

        # izris premic  ##########

        group_line = QtWidgets.QGroupBox("Iskanje premic", self)

        self.group_radiobuttons = QtWidgets.QButtonGroup(self)
        self.oneline = QtWidgets.QRadioButton()
        self.oneline.setChecked(True)
        self.oneline.setText("Najmočnejša premica")
        self.multiline = QtWidgets.QRadioButton()
        self.multiline.setText("Premice s številom presečišč vsaj")
        self.group_radiobuttons.addButton(self.oneline, 1)
        self.group_radiobuttons.addButton(self.multiline, 2)
        self.group_radiobuttons.buttonToggled.connect(self.toggleInputNumlines)

        hbox_multiline = QtWidgets.QHBoxLayout()
        self.input_numlines = QtWidgets.QLineEdit(self)
        self.input_numlines.setValidator(QtGui.QIntValidator())
        self.input_numlines.setPlaceholderText("Vnesi število")
        self.input_numlines.setDisabled(True)
        hbox_multiline.addWidget(self.multiline)
        hbox_multiline.addWidget(self.input_numlines)

        self.search_lines = QtWidgets.QPushButton("Poišči premice", self)
        self.clear_lines = QtWidgets.QPushButton("Počisti premice", self)

        vbox4 = QtWidgets.QVBoxLayout()
        vbox4.addWidget(self.oneline)
        vbox4.addLayout(hbox_multiline)
        vbox4.addWidget(self.search_lines)
        vbox4.addWidget(self.clear_lines)
        group_line.setLayout(vbox4)

        ##########

        # združitev vse grup  ##########

        vbox4 = QtWidgets.QVBoxLayout()
        vbox4.addWidget(group_loading_images)
        vbox4.addWidget(group_edges)
        vbox4.addWidget(group_hough)
        vbox4.addWidget(group_line)
        vbox4.addStretch()

        self.setLayout(vbox4)

        ##########

    def clearInputs(self):
        self.input_x.clear()
        self.input_y.clear()
        self.input_std.clear()
        self.input_sprag.clear()
        self.input_zprag.clear()
        self.input_r.clear()
        self.input_fi.clear()
        self.input_numlines.clear()
        self.input_numlines.clear()
        self.oneline.setChecked(True)

    def toggleInputNumlines(self, radiobutton, checked):

        if radiobutton == self.oneline and checked:
            self.input_numlines.setDisabled(True)
        elif radiobutton == self.multiline and checked:
            self.input_numlines.setEnabled(True)


class ui_ImageView(QtWidgets.QWidget):
    def __init__(self, parent=None):

        super().__init__()
        self.setMinimumWidth(1000)

        self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.ax = self.static_canvas.figure.subplots()

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(NavigationToolbar(self.static_canvas, self))
        vbox.addWidget(self.static_canvas)

    def imshow(self, iImage, gridX=None, gridY=None, xlabel=None, ylabel=None, vmax=255):

        self.static_canvas.figure.clear()
        self.ax = self.static_canvas.figure.subplots()

        if gridX is not None and gridY is not None:
            stepX = gridX[1] - gridX[0]
            stepY = gridY[1] - gridY[0]
            extent = (
                gridX[0] - 0.5 * stepX,
                gridX[-1] + 0.5 * stepX,
                gridY[-1] + 0.5 * stepY,
                gridY[0] - 0.5 * stepY,
            )
            self.ax.imshow(
                iImage, cmap=pp.cm.gray, vmin=0, vmax=vmax, extent=extent, aspect="auto"
            )
        else:
            self.ax.imshow(iImage, cmap=pp.cm.gray)

        if xlabel is not None:
            self.ax.set_xlabel(xlabel)

        if ylabel is not None:
            self.ax.set_ylabel(ylabel)

        self.ax.figure.canvas.draw()

    def plot_point(self, x, y, x_idx, y_idx, iImage_acc):

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=8,
            markeredgecolor="red",
            markerfacecolor="None",
        )
        title = "; ".join(
            [
                f"{i+1}.: \u03C6: {x[i]:.1f}, r: {y[i]:.1f}, N: {int(iImage_acc[y_idx[i], x_idx[i]])}"
                for i in range(len(x))
            ]
        )
        self.ax.set_title(title, fontsize=9)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.figure.canvas.draw()

    def plot_line(self, x, y):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.plot(x, y, color="red")

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.figure.canvas.draw()

    def clear(self):
        self.static_canvas.figure.clear()
        self.ax = self.static_canvas.figure.subplots()
        self.ax.figure.canvas.draw()


class mainWindow(QtWidgets.QWidget):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Vaja 8: Houghova preslikava")

        self.rightPanel = ui_RightPanel(self)

        self.view_original = ui_ImageView(self)
        self.view_edges = ui_ImageView(self)
        self.view_hough = ui_ImageView(self)

        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(self.view_original, "Vhodna slika")
        tabs.addTab(self.view_edges, "Slika robov")
        tabs.addTab(self.view_hough, "Houghova preslikava")

        hbox = QtWidgets.QHBoxLayout(self)
        hbox.addWidget(tabs)
        hbox.addWidget(self.rightPanel)

        self.iImage = None
        self.iImage_edge = None
        self.iImage_acc = None

        self.rightPanel.clear_button.clicked.connect(self.clearInputs)
        self.rightPanel.load_button.clicked.connect(self.loadImage)
        self.rightPanel.edges_button.clicked.connect(self.findEdges)
        self.rightPanel.hough_button.clicked.connect(self.execHough)
        self.rightPanel.search_lines.clicked.connect(self.findLines)
        self.rightPanel.clear_lines.clicked.connect(self.clearLines)

    def clearInputs(self):
        self.view_original.clear()
        self.view_edges.clear()
        self.view_hough.clear()
        self.rightPanel.clearInputs()

        self.iImage = None
        self.iImage_edge = None
        self.iImage_acc = None

        self.accAxes = None
        self.lineParams = None

    def loadImage(self):
        iPath = QtWidgets.QFileDialog.getOpenFileName(
            caption="Izberi slike...", filter="Images (*.raw *.png)"
        )[0]

        if not os.path.isfile(iPath):
            return

        else:
            try:
                if iPath.split(".")[-1] == "raw":
                    if (
                        not self.rightPanel.input_x.text()
                        or not self.rightPanel.input_y.text()
                    ):
                        warning("Pri nalaganju slike je prišlo do težave.")
                    else:
                        self.iImage = loadImage(
                            iPath,
                            (
                                int(self.rightPanel.input_x.text()),
                                int(self.rightPanel.input_y.text()),
                            ),
                            numpytypes[self.rightPanel.input_bits.currentText()],
                        )
                elif iPath.split(".")[-1] == "png":
                    self.iImage = pp.imread(iPath)

            except TypeError:
                warning("Pri nalaganju slike je prišlo do težave.")
                return

            self.view_original.imshow(self.iImage)

            if self.iImage_edge is not None:
                self.view_edges.clear()
                self.iImage_edge = None

            if self.iImage_acc is not None:
                self.view_hough.clear()
                self.iImage_acc = None

    def findEdges(self):

        try:
            low_iThreshold = float(self.rightPanel.input_sprag.text().replace(",", "."))
            high_iThreshold = float(
                self.rightPanel.input_zprag.text().replace(",", ".")
            )
            iStd = float(self.rightPanel.input_std.text().replace(",", "."))
        except ValueError:
            warning("Pri iskanju robov je prišlo do težave.")
            return

        if low_iThreshold > 1 or low_iThreshold < 0:
            warning("Spodnji prag mora biti med 0 in 1")
            return

        if high_iThreshold > 1 or high_iThreshold < 0:
            warning("Zgornji prag mora biti med 0 in 1")
            return

        if iStd < 0:
            warning("Standardni odklon mora biti pozitiven.")
            return

        if self.iImage is None:
            warning("Originalna slika ne obstaja.")
        else:
            self.clearLines()

            self.iImage_edge = findEdgesCanny(
                self.iImage, low_iThreshold, high_iThreshold, iStd
            )

            self.view_edges.imshow(self.iImage_edge)

            if self.iImage_acc is not None:
                self.view_hough.clear()
                self.iImage_acc = None

    def execHough(self):

        try:
            stepR = float(self.rightPanel.input_r.text().replace(",", "."))
            stepF = (
                float(self.rightPanel.input_fi.text().replace(",", "."))
                + np.finfo(np.float16).eps
            )
        except ValueError:
            warning("Pri Houghovi preslikavi je prišlo do težave.")
            return

        if self.iImage_edge is None:
            warning("Slika robov ne obstaja.")
        else:
            self.clearLines()

            self.iImage_acc, rangeR, rangeF = houghTransform2D2P(
                self.iImage_edge, stepR, stepF
            )

            self.accAxes = {"rangeR": rangeR, "rangeF": rangeF}

            self.view_hough.imshow(
                self.iImage_acc, rangeF, rangeR, "\u03C6 [deg]", "r [px]", vmax=self.iImage_acc.max()
            )

    def findLines(self):

        if self.rightPanel.multiline.isChecked():
            try:
                iIntersect = float(
                    self.rightPanel.input_numlines.text().replace(",", ".")
                )
            except ValueError:
                warning("Pri iskanju premic je prišlo do težave.")
                return
        else:
            iIntersect = 0

        if self.iImage_acc is None:
            warning("Houghova preslikava še ni bila izvedena.")
        else:
            self.clearLines()

            maxR, maxF, maxRIdx, maxFIdx = findLocalMaxima(
                self.iImage_acc,
                self.accAxes["rangeR"],
                self.accAxes["rangeF"],
                iIntersect,
                self.rightPanel.oneline.isChecked(),
            )

            self.view_hough.plot_point(maxF, maxR, maxFIdx, maxRIdx, self.iImage_acc)

            x = np.arange(self.iImage.shape[1])
            for i in range(maxR.size):
                fi = np.deg2rad(maxF[i])
                if np.sin(fi) != 0:
                    self.view_original.plot_line(
                        x, (maxR[i] - x * np.cos(fi)) / np.sin(fi)
                    )
                    self.view_edges.plot_line(
                        x, (maxR[i] - x * np.cos(fi)) / np.sin(fi)
                    )

    def clearLines(self):

        self.view_original.clear()
        self.view_edges.clear()
        self.view_hough.clear()

        if self.iImage is not None:
            self.view_original.imshow(self.iImage)

        if self.iImage_edge is not None:
            self.view_edges.imshow(self.iImage_edge)

        if self.iImage_acc is not None:
            self.view_hough.imshow(
                self.iImage_acc,
                self.accAxes["rangeF"],
                self.accAxes["rangeR"],
                "\u03C6 [deg]",
                "r [px]",
            )


def loadImage(iPath, iSize, iType):

    """
    Naloži sliko.
    """

    fid = open(iPath, "rb")

    oImage = np.ndarray((iSize[1], iSize[0]), dtype=iType, buffer=fid.read(), order="F")

    fid.close()

    return oImage


from skimage import __version__ as sk_ver
from skimage import feature


def findEdgesCanny(iImage, low_iThreshold, high_iThreshold, iStdDev):
    
    low_iThreshold_orig = low_iThreshold
    high_iThreshold_orig = high_iThreshold

    # preveri verzijo scikit-image
    if int(sk_ver[2:4]) < 17:
        # v izogib problemom pri slikah tipa float (np.iinfo deluje samo za integerje)
        try:
            low_iThreshold = np.iinfo(iImage.dtype).max * low_iThreshold
            high_iThreshold = np.iinfo(iImage.dtype).max * high_iThreshold
        except:
            low_iThreshold = iImage.max() * low_iThreshold
            high_iThreshold = iImage.max() * high_iThreshold

    # v izogib problemom pri slikah z 'ne-crnim' ozadjem (lahko najde robove ob robovih slike)
    border = 10
    iImage = np.pad(iImage, border, mode="edge")

    # izvede iskanje robov
    try:
        oImage = feature.canny(
            iImage,
            sigma=iStdDev,
            low_threshold=low_iThreshold,
            high_threshold=high_iThreshold,
            use_quantiles=True,
        )
    except ValueError:
        oImage = feature.canny(
            iImage,
            sigma=iStdDev,
            low_threshold=low_iThreshold_orig,
            high_threshold=high_iThreshold_orig,
            use_quantiles=True,
        )

    return oImage[border:-border, border:-border]


def findLocalMaxima(iAcc, rangeR, rangeF, iIntersect, maxIntersectFlag):

    # logično podatkovno polje, kamor zapisujemo lokacije maksimumov
    accFlag = np.zeros_like(iAcc)

    # iskanje globalnega maksimuma akumulatorja
    if maxIntersectFlag:
        rIdx, fIdx = np.unravel_index(iAcc.argmax(), iAcc.shape)
        accFlag[rIdx, fIdx] = 1

    # iskanje lokalnih maksimumov akumulatorja
    else:
        # okolica iskanja
        n = 1

        # povečaj akumulator za izbrano okolico iskanja "n"
        oAcc = np.pad(iAcc, n, mode="symmetric")

        for rIdx in range(rangeR.size):
            for fIdx in range(rangeF.size):

                area = oAcc[rIdx : rIdx + 2 * n + 1, fIdx : fIdx + 2 * n + 1]

                # če je središčni maksimalen, je kandidat za lokalni maksimum
                if area[n, n] == area.max():
                    accFlag[rIdx, fIdx] = 1

    # poišči točke v akumulatorju z najmanj iIntersect premicami
    rIdx, fIdx = np.where(accFlag == 1)
    i = np.where(iAcc[(rIdx, fIdx)] >= iIntersect)[0]

    maxR = rangeR[rIdx[i]]
    maxF = rangeF[fIdx[i]]
    maxRIdx = rIdx[i]
    maxFIdx = fIdx[i]

    return maxR, maxF, maxRIdx, maxFIdx


def houghTransform2D2P(iImage, stepR, stepF):
    """
    Houghova transformacija v 2D za 2 parametra.
    """

    Y, X = iImage.shape     # Dimenzije slike

    D = np.sqrt((X-1) ** 2 + (Y-1) ** 2)    # Diagonala slike
    d = stepR * np.ceil(D / stepR)    # Dolžina vektorja r

    rangeR = np.arange(-d, d + stepR, stepR)    # Obseg vrednosti r
    rangeF = np.arange(-90, 90, stepF)    # Obseg vrednosti fi

    rangeFrad = np.deg2rad(rangeF)    # Pretvorba vrednosti fi v radiane

    idxF = np.arange(rangeF.size)    # Indeksi vrednosti fi
    oAcc = np.zeros((rangeR.size, rangeF.size))    # Akumulator
    
    for y in range(Y):
        for x in range(X):
            if iImage[y, x] > 0:
                for f_idx in idxF:
                    fi = rangeFrad[f_idx]
                    r = x * np.cos(fi) + y * np.sin(fi)

                    if rangeR[0] <= r <= rangeR[-1]:
                        diffR = np.abs(rangeR - r)
                        r_idx = diffR.argmin()

                        oAcc[r_idx, f_idx] += 1


    return oAcc, rangeR, rangeF


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = mainWindow()
    gui.show()
    sys.exit(app.exec())
