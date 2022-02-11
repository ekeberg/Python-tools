#!/usr/bin/env python
import sys
from eke import sphelper
# from PyQt4.QtCore import *
# from PyQt4.QtGui import *
from PyQt5 import QtCore, QtWidgets
import numpy
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.figure import Figure
import argparse


class AppForm(QtWidgets.QMainWindow):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter autocorrelation")
        self.a = 100
        self.ax = None
        self.create_main_frame(filename)

    def create_main_frame(self, filename):
        self.main_frame = QtWidgets.QWidget()
        self.image = None

        self.image, self.image_center = sphelper.import_spimage(
            filename, ["image", "image_center"])

        x_array = numpy.arange(self.image.shape[0]) - self.image_center[0]
        y_array = numpy.arange(self.image.shape[1]) - self.image_center[1]
        X_array, Y_array = numpy.meshgrid(x_array, y_array)
        X_array = numpy.transpose(X_array)
        Y_array = numpy.transpose(Y_array)
        self.r = numpy.sqrt(X_array**2 + Y_array**2)

        ft = numpy.fft.fft2(self.image)
        self.auto_unfiltered = numpy.fft.fftshift(abs(ft))

        self.dpi = 100
        self.fig = Figure((10.0, 10.0), dpi=self.dpi)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self.main_frame)

        self.axes = self.fig.add_subplot(111)

        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self.main_frame)

        self.a_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.a_slider.setMinimum(1)
        self.a_slider.setMaximum(300)
        self.a_slider.setValue(self.a)
        self.a_slider.setTracking(False)

        self.a_label = QtWidgets.QLabel("a = %g" % self.a)

        self.a_slider.sliderMoved.connect(self.a_changed)
        self.a_slider.valueChanged.connect(self.update_image)

        self.filter_box = QtWidgets.QCheckBox("Use filter")

        self.filter_box.stateChanged.connect(self.box_state_changed)

        vbox1 = QtWidgets.QVBoxLayout()
        vbox1.addWidget(self.canvas)
        vbox1.addWidget(self.mpl_toolbar)

        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(self.a_label)
        vbox2.addWidget(self.a_slider)
        vbox2.addWidget(self.filter_box)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        self.main_frame.setLayout(hbox)
        self.setCentralWidget(self.main_frame)
        self.update_image()

    def draw(self):
        if self.ax:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = [0, self.image.shape[0]]
            ylim = [0, self.image.shape[1]]
        if self.filter_box.checkState():
            self.out = self.auto
        else:
            self.out = self.auto_unfiltered
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(numpy.log(self.out), origin='lower')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw()

    def calculate_kernel(self):
        self.kernel = ((self.r / 2 / self.a)**4
                       * numpy.exp(2 - self.r**2 / 2 / self.a**2))
        outer_mask = self.r > 2.*self.a
        self.kernel[outer_mask] = 1

    def calculate_auto(self):
        ft = numpy.fft.fft2(self.image*self.kernel)
        self.auto = numpy.fft.fftshift(abs(ft))

    def update_image(self, new_a=None):
        if self.filter_box.checkState():
            if new_a:
                self.a_changed(new_a)
            self.calculate_kernel()
            self.calculate_auto()
        self.draw()

    def a_changed(self, new_a):
        self.a = new_a
        self.a_label.setText(f"a = {self.a}")

    def box_state_changed(self):
        self.update_image()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    form = AppForm(args.file)
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
