# before running do
# export QT_API=pyside

import sys
import os
import matplotlib
matplotlib.use("Qt4Agg")
from PySide.QtCore import *
from PySide.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from optparse import OptionParser
import pylab
import colorsys

class PhaseColorscale(object):
    def __init__(self):
        self._saturation = 1.
    def convert_pixel(self, value):
        """value is complex with amplitude in [0,1]"""
        return colorsys.hls_to_rgb(pylab.angle(value)/(2.*pylab.pi), abs(value), self._saturation)
    
    def convert_image(self, image):
        return_image = pylab.zeros((pylab.shape(image)[0], pylab.shape(image)[1], 3))
        image_max = max(image.flatten())
        for x in range(pylab.shape(image)[0]):
            for y in range(pylab.shape(image)[1]):
                return_image[x,y,:] = self.convert_pixel(image[x,y]/image_max)
        return return_image
                
        

class ImageHandler(object):
    def __init__(self):
        self._image_side = 100
        self._object_side = 10
        self._true_object = pylab.fftshift(self.random_object(self._image_side, self._object_side))
        self._fourier_space = pylab.zeros((self._image_side, self._image_side), dtype='complex128')
        self._fourier_space[:,:] = abs(pylab.fft2(self._true_object))
        self._temporary_fourier = self._fourier_space.copy()
        self._real_space = pylab.real(pylab.ifft2(self._temporary_fourier))
        self._kernel = pylab.zeros((self._image_side, self._image_side))
        self._phase = 0.

        self._plot_active_only = False

    @classmethod
    def random_object(cls, image_side, object_side):
        image = pylab.zeros((image_side, image_side))
        image[image_side/2-object_side/2:image_side/2+object_side/2,
              image_side/2-object_side/2:image_side/2+object_side/2] = pylab.random((object_side, object_side))
        return image

    def get_real(self):
        return pylab.fftshift(self._real_space)

    def get_real_plot(self):
        #return pylab.fftshift(pylab.log(self._real_space))
        self._real_space[0,0] = 0.
        return pylab.fftshift(self._real_space)

    def get_fourier(self):
        return pylab.fftshift(self._temporary_fourier)

    def get_fourier_plot(self):
        return pylab.fftshift(pylab.angle(self._temporary_fourier))

    def set_kernel(self, kernel):
        """Set the area that is beeing changed"""
        self._kernel = pylab.fftshift(kernel)

    def try_phase(self, phase):
        """Change phase without the change being permanently added to the image"""
        self._phase = phase
        self._temporary_fourier = self._fourier_space * pylab.exp(-2.j*self._kernel*phase)                
        
        if self._plot_active_only:
            temp_kernel = pylab.zeros((self._image_side, self._image_side), dtype='complex128')
            index = pylab.argmax(self._kernel)
            index_2d = pylab.unravel_index(index, (self._image_side, self._image_side))
            print "phase = " + str(phase)
            temp_kernel[index_2d[0], index_2d[1]] = pylab.exp(-2.j*pylab.pi*phase)
            temp_kernel[-index_2d[0], -index_2d[1]] = pylab.exp(2.j*pylab.pi*phase)
            print "special values = " + str(temp_kernel[index_2d[0], index_2d[1]])
            print "special values = " + str(temp_kernel[-index_2d[0], -index_2d[1]])
            self._real_space = pylab.real(pylab.ifft2(temp_kernel))
            print max(pylab.angle(temp_kernel).flatten())
        else:
            self._real_space = pylab.real(pylab.ifft2(self._temporary_fourier))

    
    def apply_phase(self):
        """Apply the change that has been done through try phase"""
        #self._fourier_space[:,:] = self._temporary_fourier
        self._fourier_space *= pylab.exp(-2.j*self._kernel*self._phase)
        self._phase = 0.

    def set_plot_active_only(value):
        self._plot_active_only = value

    def get_plot_active_only():
        return self._plot_active_only

class PickedArea(object):
    def __init__(self):
        self._coordinates = (0, 0)
        self._radius = 5
        self._image_side = 100
        self._kernel = pylab.zeros((self._image_side, self._image_side))

    def set_position(self, coordinates):
        if (len(coordinates) != 2): raise ValueError("Coordinates must be len 2 vector")
        if (coordinates[0] < 0 or coordinates[1] < 0): return False
        self._coordinates = coordinates
        return True
    def get_position(self):
        return self._coordinates

    def set_radius(self, radius):
        if (radius <= 0):
            raise ValueError("Radius <= 0")
        self._radius = radius
    def get_radius(self):
        return self._radius

    def get_kernel(self):
        """Get an antisymmetric kernel (for the perfect friedel case)"""
        x = pylab.arange(self._image_side)
        y = pylab.arange(self._image_side)
        # gaussian of width radius/2
        self._kernel[:,:] = (pylab.exp(-((x-self._coordinates[0])**2 + (y[:,pylab.newaxis]-self._coordinates[1])**2)/self._radius**2) -
                             pylab.exp(-((x[::-1]-self._coordinates[0])**2 + (y[::-1][:,pylab.newaxis]-self._coordinates[1])**2)/self._radius**2))
        return self._kernel
    

class Phaser(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Phase Problme - The Game")
        self._active_area = PickedArea()
        self._image_handler = ImageHandler()
        self._colorscale_converter = PhaseColorscale()
        self.create_main_frame()
        self.redraw()

    def pixels_to_coordinates(self, pixels):
        trans = self._fourier_axis.transData.inverted()
        return trans.transform(pixels)

    def coordinates_to_pixels(self, pixels):
        trans = self._fourier_axis.transData
        return trans.transform(pixels)

    def redraw(self):
        #draw fourier image
        self._fourier_axis.clear()
        #self._fourier_axis.imshow(self._image_handler.get_fourier_plot(), origin='lower', aspect='equal')
        self._fourier_axis.imshow(self._colorscale_converter.convert_image(self._image_handler.get_fourier()),
                                  origin='lower', aspect='equal')

        #draw picked area
        circle_center = self._active_area.get_position()
        circle_radius = self._active_area.get_radius()
        circle_1 = pylab.Circle(circle_center, radius=circle_radius*0.5, fill=None)
        self._fourier_axis.add_patch(circle_1)
        circle_2 = pylab.Circle(circle_center, radius=circle_radius*1.0, fill=None)
        self._fourier_axis.add_patch(circle_2)
        self._fourier_canvas.draw()


        #draw real space
        self._real_axis.clear()
        self._real_axis.imshow(self._image_handler.get_real_plot(), origin='lower', aspect='equal')
        self._real_canvas.draw()
        

    def create_main_frame(self):
        self._main_frame = QWidget()

        self._dpi = 100
        self._canvas_size = 10. #inch
        
        self._fourier_fig = Figure((self._canvas_size, self._canvas_size), dpi=self._dpi)
        self._fourier_canvas = FigureCanvas(self._fourier_fig)
        self._fourier_canvas.setParent(self._main_frame)
        self._fourier_axis = self._fourier_fig.add_subplot(111)
        self._fourier_axis.set_xlim(0,100)
        self._fourier_axis.set_ylim(0,100)
        #self.connect(self._fourier_canvas, SIGNAL('button_press_event'), on_press)
        def _kernel_changed():
            self._image_handler.apply_phase()
            self._phase_slider.setValue(0)
            self._image_handler.set_kernel(self._active_area.get_kernel())
        def _fourier_on_mouse_click(event):
            position = self.pixels_to_coordinates((event.x, event.y))
            if not self._active_area.set_position(position):
                return
            _kernel_changed()
            self.redraw()
        self._fourier_canvas.mpl_connect('button_press_event', _fourier_on_mouse_click)
                     

        self._real_fig = Figure((self._canvas_size, self._canvas_size), dpi=self._dpi)
        self._real_canvas = FigureCanvas(self._real_fig)
        self._real_canvas.setParent(self._main_frame)
        self._real_axis = self._real_fig.add_subplot(111)

        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        def _radius_changed(new_radius):
            self._active_area.set_radius(new_radius)
            _kernel_changed()
            self.redraw()
        self._radius_slider = QSlider(Qt.Vertical)
        self._radius_slider.setMinimum(1.)
        self._radius_slider.setMaximum(100/2) #100 is image side
        self._radius_slider.setValue(self._active_area.get_radius())
        self._radius_slider.setTracking(True)
        self._radius_label = QLabel("Radius")
        radius_layout = QVBoxLayout()
        radius_layout.addWidget(self._radius_label)
        radius_layout.addWidget(self._radius_slider)
        self.connect(self._radius_slider, SIGNAL('valueChanged(int)'), _radius_changed)

        def _phase_changed(new_phase):
            self._image_handler.try_phase(new_phase/180.*pylab.pi)
            self.redraw()
        self._phase_slider = QSlider(Qt.Vertical)
        self._phase_slider.setMinimum(-180)
        self._phase_slider.setMaximum(180)
        self._phase_slider.setValue(0)
        self._phase_slider.setTracking(True)
        self._phase_label = QLabel("Phase")
        phase_layout = QVBoxLayout()
        phase_layout.addWidget(self._phase_label)
        phase_layout.addWidget(self._phase_slider)
        self.connect(self._phase_slider, SIGNAL('valueChanged(int)'), _phase_changed)

        #self.connect(self.a_slider, SIGNAL('sliderMoved(int)'), self.a_changed)
        #self.connect(self.a_slider, SIGNAL('valueChanged(int)'), self.update_image)

        hbox = QHBoxLayout()
        hbox.addWidget(self._fourier_canvas)
        hbox.addWidget(self._real_canvas)
        hbox.addLayout(radius_layout)
        hbox.addLayout(phase_layout)

        self._main_frame.setLayout(hbox)
        self.setCentralWidget(self._main_frame)

        
def main():
    parser = OptionParser(usage="%prog PATTERN")
    (options, args) = parser.parse_args()
    
    app = QApplication(sys.argv)
    phaser = Phaser()
    phaser.show()
    app.exec_()

if __name__ == "__main__":
    main()

