#!/usr/bin/env python
import sys
import os
from eke import sphelper
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import argparse

class AppForm(QMainWindow):
    def __init__(self, filename, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Filter autocorrelation")
        self.a = 100
        self.ax = None
        self.create_main_frame(filename)

    def create_main_frame(self, filename):
        self.main_frame = QWidget()
        self.image = None

        self.image, self.image_center = sphelper.import_spimage(filename, ["image", "image_center"])

        x_array = numpy.arange(self.image.shape[0]) - self.image_center[0]
        y_array = numpy.arange(self.image.shape[1]) - self.image_center[1]
        X_array, Y_array = numpy.meshgrid(x_array, y_array)
        X_array = numpy.transpose(X_array); Y_array = numpy.transpose(Y_array)
        self.r = numpy.sqrt(X_array**2 + Y_array**2)

        self.auto_unfiltered = numpy.fft.fftshift(abs(numpy.fft.fft2(self.image)))

        self.dpi = 100
        self.fig = Figure((10.0, 10.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        
        self.axes = self.fig.add_subplot(111)

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        self.a_slider = QSlider(Qt.Vertical)
        self.a_slider.setMinimum(1)
        self.a_slider.setMaximum(300)
        self.a_slider.setValue(self.a)
        self.a_slider.setTracking(False)
        
        self.a_label = QLabel("a = %g" % self.a)

        self.connect(self.a_slider, SIGNAL('sliderMoved(int)'), self.a_changed)
        self.connect(self.a_slider, SIGNAL('valueChanged(int)'), self.update_image)

        #self.filter_label = QLabel("Use filter")
        self.filter_box = QCheckBox("Use filter")

        self.connect(self.filter_box, SIGNAL('stateChanged(int)'), self.box_state_changed)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.canvas)
        vbox1.addWidget(self.mpl_toolbar)
        
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.a_label)
        vbox2.addWidget(self.a_slider)
        vbox2.addWidget(self.filter_box)
        
        hbox = QHBoxLayout()
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
        self.kernel = (self.r/2./self.a)**4*numpy.exp(2.-self.r**2/2./self.a**2)
        self.kernel[self.r > 2.*self.a] = numpy.ones(self.kernel.shape)[self.r > 2.*self.a]

    def calculate_auto(self):
        self.auto = numpy.fft.fftshift(abs(numpy.fft.fft2(self.image*self.kernel)))

    def update_image(self,new_a = None):
        if self.filter_box.checkState():
            if new_a:
                self.a_changed(new_a)
            self.calculate_kernel()
            self.calculate_auto()
        self.draw()

    def a_changed(self,new_a):
        self.a = new_a
        self.a_label.setText("a = %g" % self.a)

    def box_state_changed(self):
        self.update_image()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    form = AppForm(args.file)
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()
