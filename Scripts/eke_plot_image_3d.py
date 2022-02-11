#!/usr/bin/env python
import matplotlib
import numpy
import scipy.interpolate
from eke import sphelper
import sys
import argparse
from eke.QtVersions import QtCore, QtWidgets
import vtk
from eke import vtk_tools
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def read_image(image_file, mask):
    if mask:
        field = 'mask'
    else:
        field = 'image'
    try:
        img = sphelper.import_spimage(image_file, [field])
    except IOError:
        raise IOError(f"{image_file} is not a readable h5 image.")
    return img


class VtkWindow(QtWidgets.QMainWindow):
    def __init__(self, volume):
        super(VtkWindow, self).__init__()
        self._default_size = 800
        self.resize(self._default_size, self._default_size)
        self._volume = numpy.ascontiguousarray(volume, dtype="float32")

        self._central_widget = QtWidgets.QWidget(self)
        self._vtk_widget = QVTKRenderWindowInteractor(self._central_widget)
        self._vtk_widget.SetInteractorStyle(
            vtk.vtkInteractorStyleRubberBandPick())

        self._float_array = vtk.vtkFloatArray()
        self._float_array.SetNumberOfComponents(1)
        self._float_array.SetVoidArray(self._volume,
                                       numpy.product(self._volume.shape),
                                       1)
        self._image_data = vtk.vtkImageData()
        self._image_data.SetDimensions(*self._volume.shape)
        self._image_data.GetPointData().SetScalars(self._float_array)

        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0., 0., 0.)

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._vtk_widget)

        self._central_widget.setLayout(self._layout)
        self.setCentralWidget(self._central_widget)

    def initialize(self):
        """Initializes the vtk object. Do this after the call to show(). If
        subclassing include all the calls that need a visible window
        in this function.
        """
        self._vtk_widget.Initialize()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)


class SurfaceViewer(VtkWindow):
    def __init__(self, volume, vmin=None, vmax=None):
        super(SurfaceViewer, self).__init__(volume)
        self._value_range = (self._volume.min(), self._volume.max())

        self._SLIDER_MAXIMUM = 10000
        self._INITIAL_SLIDER_POSITION = self._SLIDER_MAXIMUM//2
        if vmin is None:
            vmin = self._volume.min()
        if vmax is None:
            vmax = self._volume.max()
        self._level_table = self._adaptive_slider_values(self._volume,
                                                         self._SLIDER_MAXIMUM,
                                                         vmin,
                                                         vmax)
        self._surface_level = self._level_table[self._INITIAL_SLIDER_POSITION]

    def initialize(self):
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

        self._level_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._level_slider.setMaximum(self._SLIDER_MAXIMUM)
        self._level_slider.valueChanged.connect(self._slider_changed)
        self._level_slider.setValue(self._INITIAL_SLIDER_POSITION)

        self._layout.addWidget(self._level_slider)
        self._vtk_widget.Render()

    def _slider_changed(self, level):
        self._surface_level = self._level_table[level]
        self._surface_algorithm.SetValue(0, self._surface_level)
        self._surface_algorithm.Modified()
        self._vtk_widget.Render()

    @staticmethod
    def _adaptive_slider_values(density, slider_maximum, vmin, vmax):
        unique_values = numpy.unique(numpy.sort(density.flat))
        unique_values = unique_values[(unique_values >= vmin)
                                      * (unique_values <= vmax)]

        interpolator = scipy.interpolate.interp1d(
            numpy.arange(len(unique_values)), unique_values)
        level_table = interpolator(numpy.linspace(
            0, len(unique_values)-1, slider_maximum+1))
        return level_table


class SliceViewer(VtkWindow):
    def __init__(self, volume, log=False, vmin=None, vmax=None):
        super(SliceViewer, self).__init__(volume)
        self._log = log

        if vmin is None:
            vmin = max(0., self._volume.min())
        if vmax is None:
            vmax = self._volume.max()
        self._lut = vtk_tools.get_lookup_table(
            vmin, vmax, log=self._log,
            colorscale=matplotlib.rcParams["image.cmap"])

        self._picker = vtk.vtkCellPicker()
        picker_tolerance = 0.005
        self._picker.SetTolerance(picker_tolerance)

        self._plane_1 = self._setup_plane()
        self._plane_1.SetPlaneOrientationToYAxes()
        self._plane_1.SetSliceIndex(self._volume.shape[0]//2)
        self._plane_2 = self._setup_plane()
        self._plane_2.SetPlaneOrientationToZAxes()
        self._plane_2.SetSliceIndex(self._volume.shape[1]//2)

    def _setup_plane(self):
        plane = vtk.vtkImagePlaneWidget()
        if vtk_tools.VTK_VERSION >= 6:
            plane.SetInputData(self._image_data)
        else:
            plane.SetInput(self._image_data)
        plane.UserControlledLookupTableOn()
        plane.SetLookupTable(self._lut)
        plane.DisplayTextOn()
        plane.SetPicker(self._picker)
        plane.SetLeftButtonAction(1)
        plane.SetMiddleButtonAction(2)
        plane.SetRightButtonAction(0)
        plane.SetInteractor(self._vtk_widget)
        return plane

    def initialize(self):
        super(SliceViewer, self).initialize()
        self._vtk_widget.Initialize()

        self._plane_1.SetEnabled(1)
        self._plane_2.SetEnabled(1)

        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(numpy.array(self._volume.shape)/2.)
        camera.SetPosition(self._volume.shape[0] / 2,
                           self._volume.shape[1] / 2,
                           -self._volume.shape[2] * 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="%prog <3d_file.h5>")
    parser.add_argument("filename")
    parser.add_argument("-s", "--shift", action="store_true",
                        help="Shift image.")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Plot in log scale.")
    parser.add_argument("-m", "--mask", action="store_true",
                        help="Plot mask.")
    parser.add_argument("-p", "--phase", action="store_true",
                        help="Plot phase.")
    parser.add_argument("-S", "--surface", action="store_true",
                        help="Plot surface")
    parser.add_argument("--min", type=float,
                        help="Lower limit of plot values")
    parser.add_argument("--max", type=float,
                        help="Upper limit of plot values")
    args = parser.parse_args()

    if args.filename is None:
        raise IOError("No input image")

    image = read_image(args.filename, args.mask)
    if args.shift:
        image = numpy.fft.fftshift(image)
    if numpy.iscomplexobj(image):
        if args.phase:
            image = numpy.angle(image)
        else:
            image = abs(image)

    app = QtWidgets.QApplication([args.filename])
    if args.surface:
        program = SurfaceViewer(image)
    else:
        program = SliceViewer(image, args.log, vmin=args.min, vmax=args.max)
    program.show()
    program.initialize()
    program.activateWindow()
    program.raise_()

    sys.exit(app.exec_())
