#!/bin/env python
import sys
import os
import h5py
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from PyQt4 import QtCore, QtGui
from pylab import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from optparse import OptionParser

def dummy_image():
    image_side = 512
    object_size = 32.
    image = ones((image_side, )*2, dtype="complex64")
    x = (arange(image_side) - image_side/2. + 0.5)[newaxis, :]
    y = (arange(image_side) - image_side/2. + 0.5)[:, newaxis]
    image[x**2 + y**2 < object_size**2] = 0.9 + 0.001j
    return image

def embedd_image(image, pad = None):
    average_phase = exp(1.j*angle(image[abs(image) > abs(image).mean()])).mean()
    image *= exp(-1.j*average_phase)
    #image = abs(image)*exp(1.j*angle(image))
    #embedded_image = average_phase*abs(image).max()*200.*ones(image.shape, dtype="complex64")
    if (pad == None):
        embedded_image = abs(image).max()*200.*ones(image.shape, dtype="complex64")
        embedded_image += image
    else:
        embedded_image = abs(image).max()*200.*ones((pad, )*2, dtype="complex64")
        embedded_image[pad/2-image.shape[0]/2:pad/2+image.shape[0]/2, pad/2-image.shape[0]/2:pad/2+image.shape[0]/2] += image
    return embedded_image
    
def crop_to_square(image):
    new_side = floor(min(image.shape)/2.)*2
    return image[image.shape[0]/2-new_side/2:image.shape[0]/2+new_side/2,
                 image.shape[0]/2-new_side/2:image.shape[0]/2+new_side/2]

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
        try:
            with h5py.File(filename, "r") as file_handle:
                self._image = (file_handle["real"][...] + 1.j*file_handle["imag"][...]).squeeze()
        except IOError:
            print "Must provide a readble Hawk HDF5 file."
            exit(1)
                
        self._image = crop_to_square(self._image)
        if self._crop == None:
            self._crop = min(self._image.shape)

        if self._crop > min(self._image.shape):
            self._image = embedd_image(self._image, pad=self._crop)
        else:
            self._image = embedd_image(self._image)

        #self._image = self._image_sp.image
        #self._image = dummy_image()
        self._image = embedd_image(self._image)
        #self._image = embedd_image(self._image, 512)

        self._x = fftshift((arange(self._image.shape[0])-self._image.shape[0]/2.+0.5)/self._image.shape[0])[newaxis, :]
        self._y = fftshift((arange(self._image.shape[1])-self._image.shape[1]/2.+0.5)/self._image.shape[1])[:, newaxis]
        # self._x = fftshift((arange(self._image.shape[0])+self._image.shape[0]/2.+0.5))[newaxis, :]
        # self._y = fftshift((arange(self._image.shape[1])+self._image.shape[1]/2.+0.5))[:, newaxis]

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
        #self._angle_slider.sliderMoved.connect(self._angle_changed)
        self._angle_slider.valueChanged.connect(self._angle_changed)

        self._angle_label = QtGui.QLabel("angle = %g" % (self._angle/pi*180.))
        self._angle_label.setFixedWidth(100)

        self._split_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self._split_slider.setMinimum(0)
        self._split_slider.setMaximum(self._slider_length)
        self._split_slider.setValue(self._split/self._max_split*self._slider_length)
        self._split_slider.setTracking(True)
        #self._split_slider.sliderMoved.connect(self._split_changed)
        self._split_slider.valueChanged.connect(self._split_changed)
        
        self._split_label = QtGui.QLabel("split = %g" % (self._split))
        self._split_label.setFixedWidth(100)

        self._phase_slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self._phase_slider.setMinimum(0)
        self._phase_slider.setMaximum(self._slider_length)
        self._phase_slider.setValue(self._phase/360.*self._slider_length)
        self._phase_slider.setTracking(True)
        #self._phase_slider.sliderMoved.connect(self._phase_changed)
        self._phase_slider.valueChanged.connect(self._phase_changed)
        
        self._phase_label = QtGui.QLabel("phase = %g" % (self._phase/pi*180.))
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
            #self._axes.cla()
            self._plt_image = self._axes.imshow(self._image_dic, cmap="gray", interpolation="bicubic")
        else:
            self._plt_image.set_data(self._image_dic)
        self._plt_image.set_clim(vmin=self._image_dic.min(), vmax=self._image_dic.max())
        # self._axes.cla()
        # self._axes.plot(real(self._image_dic).flatten(), imag(self._image_dic).flatten(), 'o')
        #self._plt_image.set_clim(vmin=-abs(self._image_dic).max(), vmax=abs(self._image_dic).max())
        image_median = median(self._image_dic)
        deviation = max(self._image_dic.max() - image_median, image_median - self._image_dic.min())
        self._plt_image.set_clim(vmin=image_median-deviation, vmax=image_median+deviation)
        self._canvas.draw()

    def _update_image(self,new_a = None):
        # phase_diff = self._diff_2d(self._image, self._angle)
        # phase_diff[abs(phase_diff) > abs(phase_diff)[phase_diff != 0.].mean()] = 0.
        # self._image_dic = abs(self._image)**2*phase_diff
        # self.draw()
        #my_slice = (slice(512/2-50, 512/2+50), )*2
        my_slice = (slice(512/2-256, 512/2+256), )*2

        kx = self._split*2.*pi*cos(self._angle)
        ky = self._split*2.*pi*sin(self._angle)
        T_dic = 1.+exp(1.j*(self._phase - kx*self._x - ky*self._y))
        # T_zer = ones(T_dic.shape, dtype="complex64")
        # T_zer[self._x**2 + self._y**2 < (self._split*10.)**2] = exp(1.j*self._phase)
        # T_sch = ones(T_dic.shape, dtype="complex64")
        # T_sch[self._x*cos(self._angle)+self._y*sin(self._angle) > 0.] = 0.
        T = T_dic

        #T = exp(1.j*(self._phase - kx*self._x - ky*self._y))
        #self._image_dic = real(fftshift(ifft2(T*fft2(fftshift(sqrt(abs(self._image))*exp(-1.j*angle(self._image)))))))
        
        image_in = self._image

        #image_in = exp(1.j*angle(self._image[my_slice]))

        #image_in = self._image[my_slice]

        # ft = fftshift(T[my_slice]*fft2(fftshift(sqrt(abs(self._image[my_slice]))*exp(1.j*angle(self._image[my_slice])))))

        #with upsampling
        ft = fftshift(T*fft2(fftshift(image_in)))
        self._crop
        self._upsample
        # print "crop = {0}".format(self._crop)
        # print "upsample = {0}".format(self._upsample)
        final_side = int(ceil(self._crop*self._upsample/2.)*2)
        downsampling = int((self._image.shape[0]*self._upsample)/final_side)
        # print "final_side = {0}".format(final_side)
        # print "downsampling = {0}".format(downsampling)
        
        # downsampling = 4
        # final_side = 512
        ft_big = zeros((final_side, final_side), dtype="complex64")
        #ft_big[final_side/2-ft.shape[0]/2:final_side/2+ft.shape[0]/2, final_side/2-ft.shape[1]/2:final_side/2+ft.shape[1]/2] = ft
        # ft_big[final_side/2-ft.shape[0]/2/downsampling:final_side/2+ft.shape[0]/2/downsampling,
        #        final_side/2-ft.shape[0]/2/downsampling:final_side/2+ft.shape[0]/2/downsampling] = ft[::downsampling, ::downsampling]


            


        ft_downsampled = fftshift(fftshift(ft)[ft.shape[0]/2-(ft.shape[0]/2/downsampling)*downsampling:
                                               ft.shape[0]/2+(ft.shape[0]/2/downsampling)*downsampling:downsampling,
                                               ft.shape[1]/2-(ft.shape[1]/2/downsampling)*downsampling:
                                               ft.shape[1]/2+(ft.shape[1]/2/downsampling)*downsampling:downsampling])
        # print "ft_big.shape = {0}".format(ft_big.shape)
        # print "ft_downsampled.shape = {0}".format(ft_downsampled.shape)
        if ft_downsampled.shape[0] <= ft_big.shape[0]:
            ft_big[final_side/2-ft_downsampled.shape[0]/2:final_side/2+ft_downsampled.shape[0]/2,
                   final_side/2-ft_downsampled.shape[1]/2:final_side/2+ft_downsampled.shape[1]/2] = ft_downsampled
        else:
            ft_big[:, :] = ft_downsampled[ft_downsampled.shape[0]/2-final_side/2:ft_downsampled.shape[0]/2+final_side/2,
                                          ft_downsampled.shape[1]/2-final_side/2:ft_downsampled.shape[1]/2+final_side/2]
        self._image_dic = abs(fftshift(ifft2(fftshift(ft_big))))**2

        #object_pixels = self._image_dic[abs(self._image_dic) > self._image_dic.mean()]
        #object_mean = exp(1.j*angle(object_pixels.mean())).mean()
        #self._image_dic *= exp(-1.j*arctan2(imag(object_mean), real(object_mean)))
        #self._image_dic = imag(self._image_dic)
        #self._image_dic = angle(self._image_dic)

        #self._image_dic = abs(self._image_dic)**2

        # #from paper without upsampling
        # self._image_dic = abs(fftshift(ifft2(T*fft2(fftshift(sqrt(abs(self._image))*exp(-1.j*angle(self._image)))))))**2

        # #explicit split dic
        # image1 = fftshift(ifft2(fft2(fftshift(image_in))))*exp(4.5j)
        # image2 = fftshift(ifft2((exp(1.j*(self._phase-kx*self._x-ky*self._y)))*fft2(fftshift(image_in))))
        # self._image_dic = abs(image1+image2)**2

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

    def _split_changed(self,new_split):
        #self.a = self.a_slider.getValue()
        self._split = self._max_split * float(new_split) / float(self._slider_length)
        self._split_label.setText("split = %g" % (self._split))
        self._update_image()

    def _phase_changed(self,new_phase):
        #self.a = self.a_slider.getValue()
        self._phase = 2.*pi * float(new_phase) / float(self._slider_length)
        self._phase_label.setText("phase = %g" % (self._phase/pi*180.))
        self._update_image()


def main():
    parser = OptionParser(usage="%prog image")
    parser.add_option("-c", action="store", type="int", dest="crop", default=None, help="Final size of the displayed image")
    parser.add_option("-u", action="store", type="float", dest="upsample", default=1.,
                      help="The final image is upsampled by this ratio")
    (options, args) = parser.parse_args()
    
    if len(args) == 0:
        print "Must provide an image."
        exit(1)
        
    if options.upsample < 1.:
        print "Upsample must be a positive value."
        exit(1)

    app = QtGui.QApplication(sys.argv)
    form = AppForm(args[0], options.crop, options.upsample)
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()

