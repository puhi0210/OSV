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
    raise ImportError("Qt4 backend is not available. Please use Qt5.")

import matplotlib.pyplot as pp
import numpy as np
from matplotlib.figure import Figure
import matplotlib.patches as patches


def warning(text):

    message = QtWidgets.QMessageBox()
    message.setIcon(QtWidgets.QMessageBox.Warning)
    message.setText(text)
    message.setInformativeText("Vnesite potrebne podatke.")
    message.addButton(QtWidgets.QMessageBox.Ok)
    message.exec()


numpytypes = {
    "08bit": np.uint8,
    "16bit": np.uint16,
}

min_fig_width = 400
min_fig_height = 300
min_middle_widget_width = 300
half_min_middle_widget_width = int(min_middle_widget_width / 2)


class ui_TopPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):

        super().__init__()

        # self.setFixedWidth(1000)

        # ----------------------------------------------------------
        ## izbira nacina poravnave
        group_registration = QtWidgets.QGroupBox("Poravnava", self)

        self.group_radiobuttons = QtWidgets.QButtonGroup(self)
        self.first_option = QtWidgets.QRadioButton()
        self.first_option.setChecked(True)
        self.first_option.setText("afina interpolacijska poravnava")
        self.second_option = QtWidgets.QRadioButton()
        self.second_option.setText("afina aproksimacijska poravnava")
        self.third_option = QtWidgets.QRadioButton()
        self.third_option.setText("brez poravnave (identiteta)")
        self.group_radiobuttons.addButton(self.first_option, 1)
        self.group_radiobuttons.addButton(self.second_option, 2)
        self.group_radiobuttons.addButton(self.third_option, 3)

        self.register_images = QtWidgets.QPushButton("Poravnaj sliki", self)
        self.save_gui_printscreen = QtWidgets.QPushButton("Shrani sliko GUI", self)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.first_option)
        hbox.addWidget(self.second_option)
        hbox.addWidget(self.third_option)
        hbox.addWidget(self.register_images)
        hbox.addWidget(self.save_gui_printscreen)
        group_registration.setLayout(hbox)

        h_box = QtWidgets.QHBoxLayout()
        h_box.addWidget(group_registration)
        h_box.addStretch()

        self.setLayout(h_box)


class ui_MiddleLeftPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, pts_list=[]):

        super().__init__()

        self.pts_list = pts_list

        self.setMinimumWidth(half_min_middle_widget_width)
        # ----------------------------------------------------------
        ## referencna slika
        group_reference_image = QtWidgets.QGroupBox("Referencna slika", self)
        # load image
        self.load_button_ref = QtWidgets.QPushButton("Nalozi sliko...", self)
        # list of points
        self.listWidget = QtWidgets.QListWidget(self)
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # delete item from list
        self.del_item_button = QtWidgets.QPushButton("Odstrani", self)

        # create widget
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.load_button_ref)
        vbox.addWidget(self.listWidget)
        vbox.addWidget(self.del_item_button)
        group_reference_image.setLayout(vbox)

        v_box = QtWidgets.QVBoxLayout()
        v_box.addWidget(group_reference_image)
        v_box.addStretch()

        self.setLayout(v_box)

    def _add_to_list(self, itm):
        item = QtWidgets.QListWidgetItem(str(itm))
        self.listWidget.addItem(item)
        self.pts_list.append(itm)

    def remove_selected(self):
        listItems = self.listWidget.selectedItems()
        if listItems:
            for item in listItems:
                self.listWidget.takeItem(self.listWidget.row(item))
                self.pts_list.remove(str_point_to_tuple(item.text()))


def str_point_to_tuple(inp):
    tmp = inp.replace("(", "").replace(")", "").split(",")
    return tuple([float(i) for i in tmp])


class ui_MiddleRightPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, pts_list=[]):

        super().__init__()
        self.pts_list = pts_list

        self.setMinimumWidth(half_min_middle_widget_width)
        # ----------------------------------------------------------
        ## vhodna slika
        group_input_image = QtWidgets.QGroupBox("Vhodna slika", self)
        # load image
        self.load_button_input = QtWidgets.QPushButton("Nalozi sliko...", self)
        # list of points
        self.listWidget = QtWidgets.QListWidget(self)
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # delete item from list
        self.del_item_button = QtWidgets.QPushButton("Odstrani", self)
        # create widget
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.load_button_input)
        vbox.addWidget(self.listWidget)
        vbox.addWidget(self.del_item_button)
        group_input_image.setLayout(vbox)

        v_box = QtWidgets.QVBoxLayout()
        v_box.addWidget(group_input_image)
        v_box.addStretch()

        self.setLayout(v_box)

    def _add_to_list(self, itm):
        item = QtWidgets.QListWidgetItem(str(itm))
        self.listWidget.addItem(item)
        self.pts_list.append(itm)

    def remove_selected(self):
        listItems = self.listWidget.selectedItems()
        if listItems:
            for item in listItems:
                self.listWidget.takeItem(self.listWidget.row(item))
                self.pts_list.remove(str_point_to_tuple(item.text()))


class ui_MiddleBottomPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):

        super().__init__()

        self.setMinimumWidth(min_middle_widget_width)
        self.setMinimumHeight(min_fig_height)
        # ----------------------------------------------------------
        ## poravnana vhodna slika
        # group_reg_input_image = QtWidgets.QGroupBox("Poravnana vhodna slika", self)

        # ----------------------------------------------------------
        ## racunanje napake poravnave
        group_error_calculation = QtWidgets.QGroupBox(
            "Racunanje napake poravnave", self
        )

        text_mse = QtWidgets.QLabel("Vrednost MSE:", self)
        text_R2 = QtWidgets.QLabel("Vrednost R2:", self)
        self.text_before_reg_mse = QtWidgets.QLabel("  a) pred poravnavo:", self)
        self.text_after_reg_mse = QtWidgets.QLabel("  b) po poravnavi:", self)
        self.text_before_reg_R2 = QtWidgets.QLabel("  a) pred poravnavo:", self)
        self.text_after_reg_R2 = QtWidgets.QLabel("  b) po poravnavi:", self)

        vbox_mse = QtWidgets.QVBoxLayout()
        vbox_mse.addWidget(text_mse)
        vbox_mse.addWidget(self.text_before_reg_mse)
        vbox_mse.addWidget(self.text_after_reg_mse)

        vbox_R2 = QtWidgets.QVBoxLayout()
        vbox_R2.addWidget(text_R2)
        vbox_R2.addWidget(self.text_before_reg_R2)
        vbox_R2.addWidget(self.text_after_reg_R2)

        text_region = QtWidgets.QLabel("Podrocje racunanje MSE:", self)

        text_x = QtWidgets.QLabel("X:", self)
        text_y = QtWidgets.QLabel("Y:", self)
        text_w = QtWidgets.QLabel("W:", self)
        text_h = QtWidgets.QLabel("H:", self)

        vbox_x = QtWidgets.QVBoxLayout()
        vbox_y = QtWidgets.QVBoxLayout()
        vbox_w = QtWidgets.QVBoxLayout()
        vbox_h = QtWidgets.QVBoxLayout()

        self.input_x = QtWidgets.QLineEdit(self)
        self.input_x.setValidator(QtGui.QIntValidator())
        self.input_x.setText("50")

        self.input_y = QtWidgets.QLineEdit(self)
        self.input_y.setValidator(QtGui.QIntValidator())
        self.input_y.setText("155")

        self.input_w = QtWidgets.QLineEdit(self)
        self.input_w.setValidator(QtGui.QIntValidator())
        self.input_w.setText("110")

        self.input_h = QtWidgets.QLineEdit(self)
        self.input_h.setValidator(QtGui.QIntValidator())
        self.input_h.setText("40")

        vbox_x.addWidget(text_x)
        vbox_x.addWidget(self.input_x)

        vbox_y.addWidget(text_y)
        vbox_y.addWidget(self.input_y)

        vbox_w.addWidget(text_w)
        vbox_w.addWidget(self.input_w)

        vbox_h.addWidget(text_h)
        vbox_h.addWidget(self.input_h)

        hbox_input_region = QtWidgets.QHBoxLayout()
        hbox_input_region.addLayout(vbox_x)
        hbox_input_region.addLayout(vbox_y)
        hbox_input_region.addLayout(vbox_w)
        hbox_input_region.addLayout(vbox_h)

        vbox_region = QtWidgets.QVBoxLayout()
        vbox_region.addWidget(text_region)
        vbox_region.addLayout(hbox_input_region)

        self.show_region = QtWidgets.QPushButton("Prikazi podrocje", self)
        self.show_region.setCheckable(True)  # setting checkable to true

        hbox_chessboard = QtWidgets.QHBoxLayout()
        self.show_chessboard = QtWidgets.QPushButton("Prikazi sahovnico", self)
        self.show_chessboard.setCheckable(True)  # setting checkable to true
        self.show_chessboard.setMinimumWidth(half_min_middle_widget_width)
        self.input_chessboard = QtWidgets.QLineEdit(self)
        self.input_chessboard.setValidator(QtGui.QIntValidator())
        self.input_chessboard.setText("40")
        hbox_chessboard.addWidget(self.show_chessboard)
        hbox_chessboard.addWidget(self.input_chessboard)

        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addLayout(vbox_mse)
        vbox2.addLayout(vbox_R2)
        vbox2.addLayout(vbox_region)
        vbox2.addWidget(self.show_region)
        vbox2.addLayout(hbox_chessboard)
        group_error_calculation.setLayout(vbox2)

        h_box = QtWidgets.QHBoxLayout()
        h_box.addWidget(group_error_calculation)
        h_box.addStretch()

        self.setLayout(h_box)

    def change_errors_values(self, mse, r2):
        self.text_before_reg_mse.setText(f"  a) pred poravnavo: {mse[0]}")
        self.text_after_reg_mse.setText(f"  b) po poravnavi: {mse[1]}")
        self.text_before_reg_R2.setText(f"  a) pred poravnavo: {r2[0]}")
        self.text_after_reg_R2.setText(f"  b) po poravnavi: {r2[1]}")

    # method called by button
    def change_show_chessboard_text(self):
        # if button is checked
        if self.show_chessboard.isChecked():
            # setting background color to light-blue
            self.show_chessboard.setText("Prikazi razliko")
        # if it is unchecked
        else:
            self.show_chessboard.setText("Prikazi sahovnico")

    # method called by button
    def change_show_region_text(self):
        # if button is checked
        if self.show_region.isChecked():
            # setting background color to light-blue
            self.show_region.setText("Prikazi tocke")
        # if it is unchecked
        else:
            self.show_region.setText("Prikazi podrocje")

    def clearInputs(self):
        self.input_x.clear()
        self.input_y.clear()
        self.input_w.clear()
        self.input_h.clear()
        self.input_chessboard.clear()
        self.first_option.setChecked(True)


class ui_ImageView(QtWidgets.QWidget):
    def __init__(self, parent=None, pts=[], name="", add_fcn=None):

        super().__init__()
        self.setMinimumWidth(min_fig_width)
        self.setMinimumHeight(min_fig_height)
        self.pts = pts
        self.name = name
        self.add_fcn = add_fcn
        self.scatter_plot_object = []
        self.rect = None

        self.static_canvas = FigureCanvas(Figure(figsize=(2.5, 2)))
        self.ax = self.static_canvas.figure.subplots()

        NUM_COLORS = 10
        cm = pp.get_cmap("tab10")
        self.colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]

        vbox = QtWidgets.QVBoxLayout(self)
        # vbox.addWidget(NavigationToolbar(self.static_canvas, self))
        vbox.addWidget(self.static_canvas)

    def imshow(self, iImage, gridX=None, gridY=None, xlabel=None, ylabel=None):

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
                iImage, cmap=pp.cm.gray, vmin=0, vmax=255, extent=extent, aspect="auto"
            )
        else:
            self.ax.imshow(iImage, cmap=pp.cm.gray, vmin=0, vmax=255)
        self.ax.set_title(self.name, fontsize=8)

        if xlabel is not None:
            self.ax.set_xlabel(xlabel)

        if ylabel is not None:
            self.ax.set_ylabel(ylabel)

        self.ax.figure.canvas.draw()
        self.init_point_plot()
        if self.name in ["Referencna slika", "Vhodna slika"]:
            self.ax.figure.canvas.mpl_connect("button_press_event", self.onclick)

    def plot_rect(self, iArea):
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (iArea[0], iArea[1]),
            iArea[2],
            iArea[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        self.rect = self.ax.add_patch(rect)
        self.ax.figure.canvas.draw()

    def remove_rect(self):
        if self.rect is not None:
            self.rect.remove()
            self.ax.figure.canvas.draw()

    def plot_landmarks(self, pts):
        self.update_points(pts=pts)

    # Simple mouse click function to store coordinates
    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        point = tuple(np.array([ix, iy]).round(1).tolist())
        self.add_fcn(point)
        self.update_points(pts=np.array(self.pts))

    def init_point_plot(self):

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.scatter_plot_object = self.ax.scatter(
            [],
            [],
            s=18,
            marker="x",
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.figure.canvas.draw()

    def update_points(self, pts):

        if len(pts) == 0:
            return
        else:
            self.scatter_plot_object.set_offsets(pts)
            self.scatter_plot_object.set_color(self.colors[: pts.shape[0]])
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
        self.iImage_ref_pts = []
        self.iImage_input_pts = []

        self.setWindowTitle("Vaja 9: Geometrijska poravnava slik")

        self.upperPanel = ui_TopPanel(self)
        self.middleLeftPanel = ui_MiddleLeftPanel(self, pts_list=self.iImage_ref_pts)
        self.middleRightPanel = ui_MiddleRightPanel(
            self, pts_list=self.iImage_input_pts
        )
        self.bottomPanel = ui_MiddleBottomPanel(self)

        self.view_reference_img = ui_ImageView(
            self,
            name="Referencna slika",
            pts=self.iImage_ref_pts,
            add_fcn=self.middleLeftPanel._add_to_list,
        )
        self.view_input_img = ui_ImageView(
            self,
            name="Vhodna slika",
            pts=self.iImage_input_pts,
            add_fcn=self.middleRightPanel._add_to_list,
        )
        self.view_reg_img = ui_ImageView(self, name="Poravnana vhodna slika")
        self.view_diff_img = ui_ImageView(
            self, name="Razlika med referenco in poravnano vhodno sliko"
        )

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.upperPanel, alignment=QtCore.Qt.AlignCenter)

        hbox_middle = QtWidgets.QHBoxLayout(self)
        hbox_middle.addWidget(self.view_reference_img)
        hbox_middle.addWidget(self.middleLeftPanel, alignment=QtCore.Qt.AlignLeft)
        hbox_middle.addWidget(self.middleRightPanel, alignment=QtCore.Qt.AlignRight)
        hbox_middle.addWidget(self.view_input_img)
        vbox.addLayout(hbox_middle)

        hbox_bottom = QtWidgets.QHBoxLayout(self)
        hbox_bottom.addWidget(self.view_reg_img)
        hbox_bottom.addWidget(self.bottomPanel, alignment=QtCore.Qt.AlignCenter)
        hbox_bottom.addWidget(self.view_diff_img)
        vbox.addLayout(hbox_bottom)

        self.iImage_ref = None
        self.iImage_input = None
        self.iImage_reg = None
        self.iImage_diff = None
        self.iImage_chess = None

        self.upperPanel.register_images.clicked.connect(self.register_images)
        self.upperPanel.save_gui_printscreen.clicked.connect(self.make_sreenshot)
        self.middleLeftPanel.load_button_ref.clicked.connect(
            lambda: self.loadImage(reference_img=True)
        )
        self.middleLeftPanel.del_item_button.clicked.connect(
            self.middleLeftPanel.remove_selected
        )
        self.middleLeftPanel.del_item_button.clicked.connect(
            lambda: self.view_reference_img.update_points(np.array(self.iImage_ref_pts))
        )
        self.middleRightPanel.load_button_input.clicked.connect(
            lambda: self.loadImage(input_img=True)
        )
        self.middleRightPanel.del_item_button.clicked.connect(
            self.middleRightPanel.remove_selected
        )
        self.middleRightPanel.del_item_button.clicked.connect(
            lambda: self.view_input_img.update_points(np.array(self.iImage_input_pts))
        )
        self.bottomPanel.show_region.clicked.connect(self.toggle_region_points)
        self.bottomPanel.show_region.clicked.connect(
            self.bottomPanel.change_show_region_text
        )
        self.bottomPanel.show_chessboard.clicked.connect(self.toggle_chess_diff)
        self.bottomPanel.show_chessboard.clicked.connect(
            self.bottomPanel.change_show_chessboard_text
        )

    def clearInputs(self):
        self.view_reference_img.clear()
        self.view_input_img.clear()
        self.view_reg_img.clear()
        self.upperPanel.clearInputs()

        self.iImage_ref = None
        self.iImage_input = None
        self.iImage_reg = None

        self.accAxes = None
        self.lineParams = None

    def loadImage(self, reference_img=False, input_img=False):

        assert (reference_img and (not input_img)) or (
            (not reference_img) and input_img
        )

        iPath = QtWidgets.QFileDialog.getOpenFileName(
            caption="Izberi slike...", filter="Images (*.raw *.png *.bmp)"
        )[0]

        if not os.path.isfile(iPath):
            return

        else:
            try:
                if iPath.split(".")[-1] == "raw":
                    a = iPath.split("/")[-1]
                    X, Y = np.array(a.split("-")[-2].split("x"), dtype=int)
                    bits = a.split("-")[-1].split(".raw")[0]
                    iType = numpytypes[bits]
                    image = loadImage(
                        iPath=iPath,
                        iSize=(
                            X,
                            Y,
                        ),
                        iType=iType,
                    )
                elif iPath.split(".")[-1] in ["png", "bmp"]:
                    image = pp.imread(iPath)

            except TypeError:
                warning("Pri nalaganju slike je prislo do tezave.")
                return
            if reference_img:
                self.iImage_ref = image
                self.view_reference_img.imshow(self.iImage_ref)

            if input_img:
                self.iImage_input = image
                self.view_input_img.imshow(self.iImage_input)

            if self.iImage_reg is not None:
                self.view_reg_img.clear()
                self.iImage_reg = None

            if self.iImage_diff is not None:
                self.view_diff_img.clear()
                self.iImage_diff = None

    def make_sreenshot(self):
        oPath = QtWidgets.QFileDialog.getSaveFileName(
            caption="Ciljna lokacija slike...", filter="Images (*.jpg *.png *.bmp)"
        )[0]

        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.winId())
        screenshot.save(oPath, oPath.split(".")[-1])

    def register_images(self):
        if self.iImage_ref is None or self.iImage_input is None:
            warning("Pred poravnavo nalozi referencno in vhodno sliko.")
            return

        if len(self.iImage_ref_pts) != len(self.iImage_input_pts):
            warning(
                "Stevilo kontrolnih tock na referencni sliki mora biti enako stevilu kontrolnih tock na vhodni sliki."
            )
            return

        if self.upperPanel.first_option.isChecked():
            iType = "interpolation"
            if len(self.iImage_ref_pts) != 3 or len(self.iImage_input_pts) != 3:
                warning(
                    "Pri afini interpolacijski poravnavi je potrebno izbrati natancno 3 pare kontrolnih tock na referencni in vhodni sliki."
                )
                return
        elif self.upperPanel.second_option.isChecked():
            iType = "approximation"
            if len(self.iImage_ref_pts) < 4 or len(self.iImage_input_pts) < 4:
                warning(
                    "Pri afini aproksimacijski poravnavi je potrebno izbrati vsaj 4 pare kontrolnih tock na referencni in vhodni sliki."
                )
                return
        elif self.upperPanel.third_option.isChecked():
            iType = "no_registration"
        else:
            warning("Pri poravnavi je prislo do tezave.")
            return

        self.oT, self.oCP, self.iImage_reg = affineRegistration(
            iType=iType,
            iImage=self.iImage_input,
            rCP=np.array(self.iImage_ref_pts).reshape(-1, 2),
            iCP=np.array(self.iImage_input_pts).reshape(-1, 2),
        )
        if self.iImage_diff is not None or self.iImage_chess is not None:
            self.view_diff_img.clear()
            self.iImage_diff = None
            self.iImage_chess = None

        self.get_diff_and_chess()
        self.get_mse_and_r2()

        self.view_reg_img.imshow(self.iImage_reg)
        self.view_diff_img.imshow(self.iImage_diff)

    def get_diff_and_chess(self):
        try:
            step = int(self.bottomPanel.input_chessboard.text())
        except ValueError:
            warning("Pri izracunu sahovnice prislo do tezave.")
            return

        try:
            self.iImage_diff, self.iImage_chess = getImages(
                iImage1=self.iImage_ref, iImage2=self.iImage_reg, iStep=step
            )
        except ValueError:
            warning("V funkciji getImages je prislo do tezave.")
            return

    def get_iArea(self):
        try:
            x = int(self.bottomPanel.input_x.text())
            y = int(self.bottomPanel.input_y.text())
            w = int(self.bottomPanel.input_w.text())
            h = int(self.bottomPanel.input_h.text())
            self.iArea = [x, y, w, h]
        except ValueError:
            warning("Pri izracunu sahovnice prislo do tezave.")
            return

    def get_mse_and_r2(self):
        self.get_iArea()
        try:
            self.R2, self.MSE = computeError(
                rCP=np.array(self.iImage_ref_pts),
                iCP=np.array(self.iImage_input_pts),
                oCP=self.oCP,
                rImage=self.iImage_ref,
                iImage=self.iImage_input,
                oImage=self.iImage_reg,
                iArea=self.iArea,
            )
        except:
            self.R2, self.MSE = ["?", "?"], ["?", "?"]
        self.bottomPanel.change_errors_values(mse=self.MSE, r2=self.R2)

    def toggle_region_points(self):
        if self.iImage_diff is None or self.iImage_chess is None:
            warning("Najprej je potrebno opraviti poravnavo.")
            return
        self.get_diff_and_chess()
        self.get_mse_and_r2()
        if self.bottomPanel.show_region.text() == "Prikazi podrocje":
            self.view_diff_img.plot_landmarks(np.array([]))
            self.view_reg_img.plot_landmarks(np.array([]))
            self.get_iArea()
            self.view_diff_img.plot_rect(self.iArea)
            self.view_reg_img.plot_rect(self.iArea)
        elif self.bottomPanel.show_region.text() == "Prikazi tocke":
            self.view_diff_img.remove_rect()
            self.view_reg_img.remove_rect()
            self.view_diff_img.plot_landmarks(self.oCP)
            self.view_reg_img.plot_landmarks(self.oCP)

    def toggle_chess_diff(self):
        if self.iImage_diff is None or self.iImage_chess is None:
            warning("Najprej je potrebno opraviti poravnavo.")
            return
        self.get_diff_and_chess()
        self.get_mse_and_r2()
        if self.bottomPanel.show_chessboard.text() == "Prikazi sahovnico":
            self.view_diff_img.imshow(self.iImage_chess)
        elif self.bottomPanel.show_chessboard.text() == "Prikazi razliko":
            self.view_diff_img.imshow(self.iImage_diff)


def loadImage(iPath, iSize, iType):

    """
    Nalozi sliko.
    """

    fid = open(iPath, "rb")

    oImage = np.ndarray((iSize[1], iSize[0]), dtype=iType, buffer=fid.read(), order="F")

    fid.close()

    return oImage


def getRadialValue(iXY, iCP):
    """
    Vrednost radialne funkcije
    iXY = [x, y]
    iCP = [[x_1, y_1],...,[x_n, y_n]]
    """

    # stevilo kontrolnih tock
    K = iCP.shape[0]
    # inicializacija vrednosti
    oValue = np.zeros(K)

    # zanka cez kontrolne tocke
    x_i = iXY[0]
    y_i = iXY[1]
    for k in range(K):
        x_k = iCP[k, 0]
        y_k = iCP[k, 1]
        # razdalja med tocko in kontrolno tocko
        r = np.sqrt((x_i - x_k) * 2 + (y_i - y_k) * 2)
        # vrednost radialne funkcije
        if r > 0:
            oValue[k] = -(r ** 2) * np.log(r)
    return oValue


def transformImage(iType, iImage, iDim, iP, iBgr=0, iInterp=0):
    """
    Geometrijska preslikava v 2D:
        - privzeta interpolacija je reda 0
    """

    # dimenzija vhodne in izhodne slike (sta enake velikosti)
    [Y, X] = iImage.shape

    # inicializacija izhodne slike
    oImage = np.ones((Y, X)) * iBgr

    # zanka cez vse slikovne elemente
    for y in range(Y):
        for x in range(X):
            # indeks slikovnega elementa -> trenutna tocka
            pt = np.array([x, y]) * iDim
            # afina geometrijska preslikava
            if iType == "affine":
                # preslikana tocka
                pt = iP @ np.append(pt, 1)
                pt = pt[:2]
            # geometrijska preslikava z radialno funkcijo
            elif iType == "radial":
                # preslikana tocka
                U = getRadialValue(pt, iP["pts"])
                pt = np.array([U @ iP["coef"][:, 0], U @ iP["coef"][:, 1]])
            # preslikana tocka -> indeks slikovnega elementa
            pt = pt / iDim
            # --------------------------------------------------------
            # interpolacija vrednosti
            if iInterp == 0:
                # A - interpolacija reda 0 (VAJA 3: interpolateImage)
                px = np.round(pt).astype(np.int64)
                # preveri veljavnost koordinate slikovnega elementa
                if px[0] < X and px[1] < Y and px[0] >= 0 and px[1] >= 0:
                    # priredi sivinsko vrednost
                    oImage[y, x] = iImage[px[1], px[0]]
            # --------------------------------------------------------
            # DOMACA NALOGA: vprasanje st. 1
            elif iInterp == 1:
                # B - interpolacija reda 1 (VAJA 3: interpolateImage)
                px = np.floor(pt).astype(np.int64)
                # preveri veljavnost koordinate slikovnega elementa
                if px[0] < X and px[1] < Y and px[0] >= 0 and px[1] >= 0:
                    # izracunaj utezi
                    a = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 1))
                    b = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 1))
                    c = abs(pt[0] - (px[0] + 1)) * abs(pt[1] - (px[1] + 0))
                    d = abs(pt[0] - (px[0] + 0)) * abs(pt[1] - (px[1] + 0))

                    # izracunaj pripadajoce sivinske vrednosti
                    # opomba: kaj se zgodi, ko je pixel zadnja tocka?
                    # utez pride 0, zato damo samo min(), da se izognemo "out of bounds"
                    sa = iImage[px[1] + 0, px[0] + 0]
                    sb = iImage[px[1] + 0, min(px[0] + 1, iImage.shape[1] - 1)]
                    sc = iImage[min(px[1] + 1, iImage.shape[0] - 1), px[0] + 0]
                    sd = iImage[
                        min(px[1] + 1, iImage.shape[0] - 1),
                        min(px[0] + 1, iImage.shape[1] - 1),
                    ]

                    # izracunaj sivinsko vrednost in jo zaokrozi
                    s = np.floor(a * sa + b * sb + c * sc + d * sd)
                    # priredi sivinsko vrednost
                    oImage[y, x] = s
    return oImage


# afina poravnava slik
def affineRegistration(iType, rCP, iCP, iImage):
    """
    Funkcije za afino poravnavo kontrolnih tock
    """

    oT = np.eye(3)

    X = rCP[:, 0]
    Y = rCP[:, 1]
    U = iCP[:, 0]
    V = iCP[:, 1]

    if iType == "interpolation":
        XY = np.ones((3, 3))
        UV = np.ones((3, 3))

        XY[0], XY[1] = X, Y
        UV[0], UV[1] = U, V
        
        if np.linalg.det(XY) == 0:
            raise ValueError("Matrika ni invertibilna. Izberi boljše kontrolne točke.")
        else:
            oT = UV @ np.linalg.inv(XY)

    if iType == "approximation":
        x = X.mean()
        y = Y.mean()
        u = U.mean()
        v = V.mean()

        xx = (X**2).mean()
        yy = (Y**2).mean()

        xy = (X * Y).mean()
        ux = (U * X).mean()
        uy = (U * Y).mean()
        vx = (V * X).mean()
        vy = (V * Y).mean()

        sub_matrix = np.array([[xx, xy, x], [xy, yy, y], [x, y, 1]]).reshape(3, 3)
        XY = np.zeros((6, 6))
        XY[:3, :3] = sub_matrix
        XY[3:, 3:] = sub_matrix

        UV = np.array([ux, uy, u, vx, vy, v])

        if np.linalg.det(XY) == 0:
            raise ValueError("Matrika ni invertibilna. Izberi boljše kontrolne točke.")
        else:
            oT_vec = np.linalg.inv(XY) @ UV
            oT[:2] = oT_vec.reshape(2, 3)



    pts = np.ones((3, U.size))
    pts[0], pts[1] = U, V
    oCP = np.linalg.inv(oT) @ pts
    oCP = oCP[:2, :].T

    oImage = transformImage(
        iType = "affine",
        iImage = iImage,
        iDim = [1, 1],
        iP = oT,
        iBgr = 0,
        iInterp = 1
        )

    return oT, oCP, oImage


# doloci slike rezultatov
def getImages(iImage1, iImage2, iStep):
    """
    Funkcije za izracun rezultatov poravnave v obliki slike razlik in slike sahovnice
    """

    diff = iImage1.astype(float) - iImage2.astype(float)
    diff -= -255
    diff /= 2
    oImage1 = diff.astype(np.uint8)

    Y, X = iImage2.shape
    oImage2 = iImage2.copy()

    for y in range(0, Y, iStep):
        for x in range(0, X, iStep):
            isDiag = (x + y) / iStep % 2
            if isDiag:
                xLim = min(x + iStep, X)
                yLim = min(y + iStep, Y)
                oImage2[y:yLim, x:xLim] = iImage2[y:yLim, x:xLim]


    return oImage1, oImage2


# napaka R2 in MSE pred in po poravnavi
def computeError(rCP, iCP, oCP, rImage, iImage, oImage, iArea):
    """
    Funkcije za izracun napake poravnave
    """

    raise NotImplementedError("Implement me")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = mainWindow()
    gui.show()
    sys.exit(app.exec())