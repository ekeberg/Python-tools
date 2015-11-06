#!/usr/bin/env python
import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)
#from pylab import *
import numpy
#import spimage
from eke import sphelper
import sys
from optparse import OptionParser
from eke.QtVersions import QtCore, QtGui
import vtk
from eke import vtk_tools
from vtk.util import numpy_support
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import time

def read_image(image_file, mask):
    if mask:
        field = 'mask'
    else:
        field = 'image'
    try:
        img = abs(sphelper.import_spimage(image_file, [field]))
    except:
        raise IOError("%s is not a readable h5 image." % image_file)
    return img


class VtkWindow(QtGui.QMainWindow):
    def __init__(self, volume):
        super(VtkWindow, self).__init__()
        self._default_size = 600
        self.resize(self._default_size, self._default_size)
        self._volume = numpy.ascontiguousarray(volume, dtype="float32")

        self._central_widget = QtGui.QWidget(self)
        self._vtk_widget = QVTKRenderWindowInteractor(self._central_widget)
        self._vtk_widget.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())

        self._float_array = vtk.vtkFloatArray()
        self._float_array.SetNumberOfComponents(1)
        self._float_array.SetVoidArray(self._volume, numpy.product(self._volume.shape), 1)
        self._image_data = vtk.vtkImageData()
        self._image_data.SetDimensions(*self._volume.shape)
        self._image_data.GetPointData().SetScalars(self._float_array)

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0., 0., 0.)

        self._layout = QtGui.QVBoxLayout()
        self._layout.addWidget(self._vtk_widget)

        self._central_widget.setLayout(self._layout)
        self.setCentralWidget(self._central_widget)

    def initialize(self):
        """Initializes the vtk object. Do this after the call to show(). If subclassing
        include all the calls that need a visible window in this function."""
        self._vtk_widget.Initialize()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)


class SurfaceViewer(VtkWindow):
    def __init__(self, volume):
        super(SurfaceViewer, self).__init__(volume)
        #self._vtk_widget.Initialize()

        self._value_range = (self._volume.min(), self._volume.max())
        self._surface_level = numpy.mean(self._value_range)

    def initialize(self):
        #self._vtk_widget.Initialize()
        super(SurfaceViewer, self).initialize()
        self._surface_algorithm = vtk.vtkMarchingCubes()
        if vtk_tools.VTK_VERSION >= 6:
            self._surface_algorithm.SetInputData(self._image_data)
        else:
            self._surface_algorithm.SetInput(self._image_data)
        self._surface_algorithm.ComputeNormalsOn()
        self._surface_algorithm.SetValue(0, self._surface_level)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._surface_algorithm.GetOutputPort())
        mapper.ScalarVisibilityOff()
        self._actor = vtk.vtkActor()
        self._actor.GetProperty().SetColor(0., 1., 0.)
        self._actor.SetMapper(mapper)
        self._renderer.AddViewProp(self._actor)
        self._renderer.Render()
        self._vtk_widget.Render()
        
        self._level_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self._SLIDER_MAXIMUM = 100
        self._level_slider.setValue(self._SLIDER_MAXIMUM/2.)
        self._level_slider.setMaximum(self._SLIDER_MAXIMUM)
        self._level_slider.valueChanged.connect(self._slider_changed)

        self._level_table = self._adaptive_slider_values(self._volume, self._SLIDER_MAXIMUM)

        self._layout.addWidget(self._level_slider)

    def _slider_changed(self, level):
        # self._surface_level = (float(level) / float(self._SLIDER_MAXIMUM) * (self._value_range[1] - self._value_range[0]) +
        #                        self._value_range[0])
        self._surface_level = self._level_table[level]
        self._surface_algorithm.SetValue(0, self._surface_level)
        self._surface_algorithm.Modified()
        self._vtk_widget.Render()

    @staticmethod
    def _adaptive_slider_values(density, slider_maximum):
        level_table = numpy.zeros(slider_maximum+1, dtype="float64")
        density_flat = density.flatten()
        for slider_level in range(slider_maximum+1):
            level_table[slider_level] = numpy.percentile(density_flat, float(slider_level) / float(slider_maximum) * 100.)
        return level_table

class SliceViewer(VtkWindow):
    def __init__(self, volume, log=False):
        super(SliceViewer, self).__init__(volume)
        self._log = log
        
    def initialize(self):
        super(SliceViewer, self).initialize()
        self._vtk_widget.Initialize()

        picker = vtk.vtkCellPicker()
        picker_tolerance = 0.005
        picker.SetTolerance(picker_tolerance)
        
        lut = vtk_tools.get_lookup_table(self._volume.min(), self._volume.max(), log=self._log,
                                         colorscale=matplotlib.rcParams["image.cmap"])

        def setup_plane():
            plane = vtk.vtkImagePlaneWidget()
            if vtk_tools.VTK_VERSION >= 6:
                plane.SetInputData(self._image_data)
            else:
                plane.SetInput(self._image_data)
            plane.UserControlledLookupTableOn()
            plane.SetLookupTable(lut)
            plane.DisplayTextOn()
            plane.SetPicker(picker)
            plane.SetLeftButtonAction(1)
            plane.SetMiddleButtonAction(2)
            plane.SetRightButtonAction(0)
            plane.SetInteractor(self._vtk_widget)
            plane.SetEnabled(1)
            return plane

        plane_1 = setup_plane()
        plane_1.SetPlaneOrientationToXAxes()
        plane_1.SetSliceIndex(self._volume.shape[0]/2)
        plane_2 = setup_plane()
        plane_2.SetPlaneOrientationToYAxes()
        plane_2.SetSliceIndex(self._volume.shape[1]/2)
        
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(numpy.array(self._volume.shape)/2.)
        camera.SetPosition(self._volume.shape[0]/2., self._volume.shape[1]/2., -self._volume.shape[2]*2.)
        

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <3d_file.h5>")
    parser.add_option("-s", action="store_true", dest="shift", help="Shift image.")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot mask.")
    parser.add_option("-S", action="store_true", dest="surface", help="Plot surface")
    (options, args) = parser.parse_args()
    
    if len(args) == 0: raise InputError("No input image")

    image = read_image(args[0], options.mask)
    if options.shift:
        image = numpy.fft.fftshift(image)
    
    #plot_image_3d(image, options.shift, options.log, options.surface)
    app = QtGui.QApplication(["plot_image_3d"])
    if options.surface:
        program = SurfaceViewer(image)
    else:
        program = SliceViewer(image, options.log)
    program.show()
    program.initialize()
    sys.exit(app.exec_())
    
