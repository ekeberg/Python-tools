#!/usr/bin/env python
import sys
import os
import h5py
from PyQt4 import QtCore, QtGui
import numpy
from eke import sphelper
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import argparse

def dummy_image():
    image_side = 512
    object_size = 32.
    image = ones((image_side, )*2, dtype="complex64")
    x = (numpy.arange(image_side) - image_side/2. + 0.5)[numpy.newaxis, :]
    y = (numpy.arange(image_side) - image_side/2. + 0.5)[:, numpy.newaxis]
    image[x**2 + y**2 < object_size**2] = 0.9 + 0.001j
    return image

def embedd_image(image, pad = None):
    average_phase = numpy.exp(1.j*numpy.angle(image[abs(image) > abs(image).mean()])).mean()
    image *= numpy.exp(-1.j*average_phase)
    if (pad == None):
        embedded_image = abs(image).max()*200.*numpy.ones(image.shape, dtype="complex64")
        embedded_image += image
    else:
        embedded_image = abs(image).max()*200.*numpy.ones((pad, )*2, dtype="complex64")
        embedded_image[pad//2-image.shape[0]//2:pad//2+image.shape[0]//2,
                       pad//2-image.shape[0]//2:pad//2+image.shape[0]//2] += image
    return embedded_image
    
def crop_to_square(image):
    new_side = numpy.floor(min(image.shape)/2.)*2
    return image[image.shape[0]//2-new_side//2:image.shape[0]//2+new_side//2,
                 image.shape[0]//2-new_side//2:image.shape[0]//2+new_side//2]

class AppForm(QtGui.QMainWindow):
    def __init__(self, filename, crop, upsample, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setWindowTitle("Differential Interference Contrast Microscopy - {0}".format(filename))
        self._crop = crop
        self._upsample = upsample
        self._angle = 0.
        self._split = 2.
        self._max_split = 5.
        self._phase = 0.
        self._plt_image = None
        self._create_actions()
        self._setup_menus()
        self._create_main_frame(filename)

    def _create_main_frame(self, filename):
        self._main_frame = QtGui.QWidget()
        self._image = None
        self._image = numpy.complex128(sphelper.import_spimage(filename))
                
        self._image = crop_to_square(self._image)
        if self._crop == None:
            self._crop = min(self._image.shape)

        if self._crop > min(self._image.shape):
            self._image = embedd_image(self._image, pad=self._crop)
        else:
            self._image = embedd_image(self._image)

        self._image = embedd_image(self._image)

        self._x = numpy.fft.fftshift((numpy.arange(self._image.shape[0])-self._image.shape[0]/2.+0.5)/float(self._image.shape[0]))[numpy.newaxis, :]
        self._y = numpy.fft.fftshift((numpy.arange(self._image.shape[1])-self._image.shape[1]/2.+0.5)/float(self._image.shape[1]))[:, numpy.newaxis]

        self._dpi = 100
        self._fig = Figure((10.0, 10.0), dpi=self._dpi)
        self._fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setParent(self._main_frame)
        
        self._axes = self._fig.add_subplot(111)
        self._axes.set_xticks([])
        self._axes.set_yticks([])

        self._mpl_toolbar = NavigationToolbar(self._canvas, self._main_frame)

        self._slider_length = 1000
        self._angle_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self._angle_slider.setMinimum(0)
        self._angle_slider.setMaximum(self._slider_length)
        self._angle_slider.setValue(self._angle/360.*self._slider_length)
        self._angle_slider.setTracking(True)
        self._angle_slider.valueChanged.connect(self._angle_changed)

        self._angle_label = QtGui.QLabel("angle = %g" % (self._angle/numpy.pi*180.))
        self._angle_label.setFixedWidth(100)

        self._split_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self._split_slider.setMinimum(0)
        self._split_slider.setMaximum(self._slider_length)
        self._split_slider.setValue(self._split/self._max_split*self._slider_length)
        self._split_slider.setTracking(True)
        self._split_slider.valueChanged.connect(self._split_changed)
        
        self._split_label = QtGui.QLabel("split = %g" % (self._split))
        self._split_label.setFixedWidth(100)

        self._phase_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self._phase_slider.setMinimum(0)
        self._phase_slider.setMaximum(self._slider_length)
        self._phase_slider.setValue(self._phase/360.*self._slider_length)
        self._phase_slider.setTracking(True)
        self._phase_slider.valueChanged.connect(self._phase_changed)
        
        self._phase_label = QtGui.QLabel("phase = %g" % (self._phase/numpy.pi*180.))
        self._phase_label.setFixedWidth(100)

        vbox1 = QtGui.QVBoxLayout()
        vbox1.addWidget(self._canvas)
        vbox1.addWidget(self._mpl_toolbar)
        
        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self._angle_label)
        vbox2.addWidget(self._angle_slider)

        vbox3 = QtGui.QVBoxLayout()
        vbox3.addWidget(self._split_label)
        vbox3.addWidget(self._split_slider)

        vbox4 = QtGui.QVBoxLayout()
        vbox4.addWidget(self._phase_label)
        vbox4.addWidget(self._phase_slider)

        hbox = QtGui.QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        hbox.addLayout(vbox4)

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
            self._plt_image = self._axes.imshow(self._image_dic, cmap="gray", interpolation="bicubic")
        else:
            self._plt_image.set_data(self._image_dic)
        self._plt_image.set_clim(vmin=self._image_dic.min(), vmax=self._image_dic.max())
        image_median = numpy.median(self._image_dic)
        deviation = max(self._image_dic.max() - image_median, image_median - self._image_dic.min())
        self._plt_image.set_clim(vmin=image_median-deviation, vmax=image_median+deviation)
        self._canvas.draw()

    def _update_image(self,new_a = None):
        my_slice = (slice(512//2-256, 512//2+256), )*2

        kx = self._split*2.*numpy.pi*numpy.cos(self._angle)
        ky = self._split*2.*numpy.pi*numpy.sin(self._angle)
        T_dic = 1.+numpy.exp(1.j*(self._phase - kx*self._x - ky*self._y))
        T = T_dic

        image_in = self._image

        #with upsampling
        ft = numpy.fft.fftshift(T*numpy.fft.fft2(numpy.fft.fftshift(image_in)))
        self._crop
        self._upsample
        final_side = int(numpy.ceil(self._crop*self._upsample/2.)*2)
        downsampling = int((self._image.shape[0]*self._upsample)//final_side)
        
        ft_big = numpy.zeros((final_side, final_side), dtype="complex64")

        ft_downsampled = numpy.fft.fftshift(numpy.fft.fftshift(ft)[ft.shape[0]//2-(ft.shape[0]/2/downsampling)*downsampling:
                                                                   ft.shape[0]//2+(ft.shape[0]/2/downsampling)*downsampling:downsampling,
                                                                   ft.shape[1]//2-(ft.shape[1]/2/downsampling)*downsampling:
                                                                   ft.shape[1]//2+(ft.shape[1]/2/downsampling)*downsampling:downsampling])
        if ft_downsampled.shape[0] <= ft_big.shape[0]:
            ft_big[final_side//2-ft_downsampled.shape[0]//2:final_side//2+ft_downsampled.shape[0]//2,
                   final_side//2-ft_downsampled.shape[1]//2:final_side//2+ft_downsampled.shape[1]//2] = ft_downsampled
        else:
            ft_big[:, :] = ft_downsampled[ft_downsampled.shape[0]//2-final_side//2:ft_downsampled.shape[0]//2+final_side//2,
                                          ft_downsampled.shape[1]//2-final_side//2:ft_downsampled.shape[1]//2+final_side//2]
        self._image_dic = abs(numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.fftshift(ft_big))))**2

        self.draw()

    @classmethod
    def _diff_2d(cls, input, direction_angle):
        raw_diff_x = numpy.exp(-1.j*numpy.angle(input[:, 1:])) - numpy.exp(-1.j*numpy.angle(input[:, :-1]))
        diff_x = (-1.+2.*(numpy.imag(raw_diff_x*numpy.exp(-1.j*numpy.angle(input[:, 1:]))) > 0.))*abs(raw_diff_x)
        diff_x_smooth = numpy.zeros(input.shape)
        diff_x_smooth[:, :-1] += diff_x
        diff_x_smooth[:, 1:] += diff_x
        diff_x_smooth[:, 1:-1] /= 2.
        raw_diff_y = numpy.exp(-1.j*numpy.angle(input[1:, :])) - numpy.exp(-1.j*numpy.angle(input[:-1, :]))
        diff_y = (-1.+2.*(numpy.imag(raw_diff_y*numpy.exp(-1.j*numpy.angle(input[1:, :]))) > 0.))*abs(raw_diff_y)
        diff_y_smooth = numpy.zeros(input.shape)
        diff_y_smooth[:-1, :] += diff_y
        diff_y_smooth[1:, :] += diff_y
        diff_y_smooth[1:-1, :] /= 2.
        return numpy.cos(direction_angle)*diff_x_smooth + numpy.sin(direction_angle)*diff_y_smooth

    def _angle_changed(self,new_angle):
        self._angle = 2.*numpy.pi * float(new_angle) / float(self._slider_length)
        self._angle_label.setText("angle = %g" % (self._angle/numpy.pi*180.))
        self._update_image()

    def _split_changed(self,new_split):
        self._split = self._max_split * float(new_split) / float(self._slider_length)
        self._split_label.setText("split = %g" % (self._split))
        self._update_image()

    def _phase_changed(self,new_phase):
        self._phase = 2.*numpy.pi * float(new_phase) / float(self._slider_length)
        self._phase_label.setText("phase = %g" % (self._phase/numpy.pi*180.))
        self._update_image()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-c", "--crop", type=int, default=None, help="Final size of the displayed image")
    parser.add_argument("-u", "--upsample", type=float, default=1., help="The final image is upsampled by this ratio")
    args = parser.parse_args()
        
    if args.upsample < 1.:
        print("Upsample must be a positive value.")
        exit(1)

    app = QtGui.QApplication(sys.argv)
    form = AppForm(args.file, args.crop, args.upsample)
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()

