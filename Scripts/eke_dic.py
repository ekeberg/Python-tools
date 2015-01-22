#!/bin/env python
import sys, os, spimage
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from PyQt4 import QtCore, QtGui
from pylab import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from optparse import OptionParser

class AppForm(QtGui.QMainWindow):
    def __init__(self, filename, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle("Filter autocorrelation")
        self._angle = 0.
        self._plt_image = None
        self._create_actions()
        self._setup_menus()
        self._create_main_frame(filename)

    def _create_main_frame(self, filename):
        self._main_frame = QtGui.QWidget()
        self._image = None
        try:
            self._image_sp = spimage.sp_image_read(filename,0)
        except:
            print "Must provide a file"
            exit(1)

        self._image = self._image_sp.image

        self._dpi = 100
        self._fig = Figure((10.0, 10.0), dpi=self._dpi)
        self._fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setParent(self._main_frame)
        
        self._axes = self._fig.add_subplot(111)
        self._axes.set_xticks([])
        self._axes.set_yticks([])

        self._mpl_toolbar = NavigationToolbar(self._canvas, self._main_frame)

        self._slider_length = 100
        self._angle_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self._angle_slider.setMinimum(0)
        self._angle_slider.setMaximum(self._slider_length)
        self._angle_slider.setValue(self._angle)
        self._angle_slider.setTracking(True)
        
        self._angle_label = QtGui.QLabel("angle = %g" % (self._angle/pi*180.))
        self._angle_label.setFixedWidth(100)

        #self.connect(self._angle_slider, SIGNAL('sliderMoved(int)'), self._angle_changed)
        self._angle_slider.sliderMoved.connect(self._angle_changed)

        vbox1 = QtGui.QVBoxLayout()
        vbox1.addWidget(self._canvas)
        vbox1.addWidget(self._mpl_toolbar)
        
        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self._angle_label)
        vbox2.addWidget(self._angle_slider)
        
        hbox = QtGui.QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        self._main_frame.setLayout(hbox)
        self.setCentralWidget(self._main_frame)
        self._update_image()

    def _create_actions(self):
        self._actions = {}

        #exit
        self._actions["exit"] = QtGui.QAction("Exit", self)
        self._actions["exit"].setShortcut("Ctrl+Q")
        self._actions["exit"].triggered.connect(exit)

        #save image
        self._actions["save image"] = QtGui.QAction("Save image", self)
        self._actions["save image"].triggered.connect(self._on_save_image)


    def _setup_menus(self):
        self._menus = {}
        self._menus["file"] = self.menuBar().addMenu("&File")
        self._menus["file"].addAction(self._actions["save image"])
        self._menus["file"].addAction(self._actions["exit"])

    def _on_save_image(self):
        file_name = str(QtGui.QFileDialog.getSaveFileName(self, "Save file"))
        self._fig.savefig(file_name, dpi=300)

    def draw(self):
        if self._plt_image == None:
            #self._axes.cla()
            self._plt_image = self._axes.imshow(self._image_dic, cmap="gray", interpolation="bicubic")
        else:
            self._plt_image.set_data(self._image_dic)
        self._plt_image.set_clim(vmin=-abs(self._image_dic).max(), vmax=abs(self._image_dic).max())
        self._canvas.draw()

    def _update_image(self,new_a = None):
        phase_diff = self._diff_2d(self._image, self._angle)
        phase_diff[abs(phase_diff) > abs(phase_diff)[phase_diff != 0.].mean()] = 0.
        self._image_dic = abs(self._image)**2*phase_diff
        self.draw()

    @classmethod
    def _diff_2d(cls, input, direction_angle):
        raw_diff_x = exp(-1.j*angle(input[:, 1:])) - exp(-1.j*angle(input[:, :-1]))
        diff_x = (-1.+2.*(imag(raw_diff_x*exp(-1.j*angle(input[:, 1:]))) > 0.))*abs(raw_diff_x)
        diff_x_smooth = zeros(input.shape)
        diff_x_smooth[:, :-1] += diff_x
        diff_x_smooth[:, 1:] += diff_x
        diff_x_smooth[:, 1:-1] /= 2.
        raw_diff_y = exp(-1.j*angle(input[1:, :])) - exp(-1.j*angle(input[:-1, :]))
        diff_y = (-1.+2.*(imag(raw_diff_y*exp(-1.j*angle(input[1:, :]))) > 0.))*abs(raw_diff_y)
        diff_y_smooth = zeros(input.shape)
        diff_y_smooth[:-1, :] += diff_y
        diff_y_smooth[1:, :] += diff_y
        diff_y_smooth[1:-1, :] /= 2.
        #return -diff_x_smooth-diff_y_smooth
        return cos(direction_angle)*diff_x_smooth + sin(direction_angle)*diff_y_smooth

    def _angle_changed(self,new_angle):
        #self.a = self.a_slider.getValue()
        self._angle = 2.*pi * float(new_angle) / float(self._slider_length)
        self._angle_label.setText("angle = %g" % (self._angle/pi*180.))
        self._update_image()

def main():
    parser = OptionParser(usage="%prog image")
    (options, args) = parser.parse_args()
    
    if len(args) == 0:
        print "Must provide an image."
        exit(1)

    app = QtGui.QApplication(sys.argv)
    form = AppForm(args[0])
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()

