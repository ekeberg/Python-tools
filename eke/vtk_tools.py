"""Functions and classes that are useful when working with VTK. Uses vtk 6."""
import vtk as _vtk
import numpy as _numpy
import scipy as _scipy
import scipy.interpolate

from .QtVersions import QtGui, QtCore, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

VTK_VERSION = _vtk.vtkVersion().GetVTKMajorVersion()

def get_lookup_table(minimum_value, maximum_value, log=False, colorscale="jet", number_of_colors=1000):
    """Returrns a vtk lookup_table based on the specified matplotlib colorscale"""
    import matplotlib
    import matplotlib.cm
    if log:
        lut = _vtk.vtkLogLookupTable()
    else:
        lut = _vtk.vtkLookupTable()
    lut.SetTableRange(minimum_value, maximum_value)
    lut.SetNumberOfColors(number_of_colors)
    lut.Build()
    for i in range(number_of_colors):
        color = matplotlib.cm.cmap_d[colorscale](float(i) / float(number_of_colors))
        lut.SetTableValue(i, color[0], color[1], color[2], 1.)
    if VTK_VERSION >= 6:
        lut.SetUseBelowRangeColor(True)
        lut.SetUseAboveRangeColor(True)
    return lut

def array_to_float_array(array_in, dtype=None):
    """Get vtkFloatArray/vtkDoubleArray from the input numpy array."""
    if dtype == None:
        dtype = array_in.dtype
    if dtype == "float32":
        float_array = _vtk.vtkFloatArray()
    elif dtype == "float64":
        float_array = _vtk.vtkDoubleArray()
    else:
        raise ValueError("Wrong format of input array, must be float32 or float64")
    if len(array_in.shape) == 2:
        float_array.SetNumberOfComponents(array_in.shape[1])
    elif len(array_in.shape) == 1:
        float_array.SetNumberOfComponents(1)
    else:
        raise ValueError("Wrong shape of array must be 1D or 2D.")
    array_contiguous = _numpy.ascontiguousarray(array_in, dtype)
    float_array.SetVoidArray(array_contiguous,
                             _numpy.product(array_contiguous.shape), 1)
    float_array._contiguous_array = array_contiguous  # Hack to keep the array from being garbage collected
    return float_array

def array_to_vtk(array_in, dtype=None):
    """Get vtkFloatArray/vtkDoubleArray from the input numpy array."""
    if dtype == None:
        dtype = _numpy.dtype(array_in.dtype)
    else:
        dtype = _numpy.dtype(dtype)
    if dtype == _numpy.float32:
        float_array = _vtk.vtkFloatArray()
    elif dtype == _numpy.float64:
        float_array = _vtk.vtkDoubleArray()
    elif dtype == _numpy.uint8:
        float_array = _vtk.vtkUnsignedCharArray()
    elif dtype == _numpy.int8:
        float_array = _vtk.vtkCharArray()
    else:
        raise ValueError("Wrong format of input array, must be one of float32, float64, uint8, int8")
    if len(array_in.shape) != 1 and len(array_in.shape) != 2:
        raise ValueError("Wrong shape: array must be 1D or 2D.")
    #float_array.SetNumberOfComponents(_numpy.product(array_in.shape))
    # if len(array_in.shape) == 2:
    #     float_array.SetNumberOfComponents(array_in.shape[1])
    # elif len(array_in.shape) == 1:
    #     float_array.SetNumberOfComponents(1)
    float_array.SetNumberOfComponents(1)
    array_contiguous = _numpy.ascontiguousarray(array_in, dtype)
    float_array.SetVoidArray(array_contiguous, _numpy.product(array_in.shape), 1)
    float_array._contiguous_array = array_contiguous  # Hack to keep the array of being garbage collected
    # if len(array_in.shape) == 2:
    #     print "set tuple to {0}".format(array_in.shape[1])
    #     #float_array.SetNumberOfTuples(array_in.shape[1])
    #     float_array.Resize(array_in.shape[1])
    #     float_array.Squeeze()
    return float_array

def array_to_image_data(array_in, dtype=None):
    """Get vtkImageData from the input numpy array."""
    array_flat = array_in.flatten()
    float_array = array_to_float_array(array_flat, dtype)
    image_data = _vtk.vtkImageData()
    image_data.SetDimensions(*(array_in.shape[::-1]))
    image_data.GetPointData().SetScalars(float_array)
    return image_data

def window_to_png(render_window, file_name, magnification=1):
    """Get a screen capture from the provided vtkRenderWindow."""
    window_to_image_filter = _vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    #window_to_image_filter.SetMagnification(magnification)
    window_to_image_filter.SetScale(magnification)
    window_to_image_filter.SetInputBufferTypeToRGBA()
    window_to_image_filter.Update()

    writer = _vtk.vtkPNGWriter()
    writer.SetFileName(file_name)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

def poly_data_to_actor(poly_data, lut):
    """Minimal function to create an actor from a poly data. This hides the mapper
    step which is sometimes useful but doesn't need to be explicitly tampered with."""
    mapper = _vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    actor = _vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor

class SphereMap(object):
    """Plot on a spherical shell."""
    def __init__(self, n):
        self._sampling_n = None
        self._points = None
        self._polygons = None
        self._vtk_points = None
        self._vtk_poly_data = None
        self._vtk_scalars = None
        self._vtk_polygons = None
        self._vtk_mapper = None
        self._vtk_actor = None
        self._set_n(n)
        self._generate_points_and_polys()
        self._generate_vtk_objects()

    def set_n(self, sampling_n):
        """Set sampling parameter n."""
        self._set_n(sampling_n)
        self._generate_points_and_polys()
        self._update_vtk_objects()

    def _set_n(self, sampling_n):
        """Set n but does not update any object."""
        sampling_n = int(sampling_n)
        if sampling_n < 4:
            raise ValueError("SphereMap does not support values of n of 3 or lower, {0} received.".format(sampling_n))
        self._sampling_n = sampling_n

    def set_values(self, values):
        """Provide new values for the coordinates given by get_coordinates()"""
        if len(values) != self._vtk_scalars.GetNumberOfTuples():
            raise ValueError("Input value array must match number of points")
        for i in range(self._vtk_scalars.GetNumberOfTuples()):
            self._vtk_scalars.SetValue(i, values[i])
        self._vtk_scalars.Modified()

    def get_number_of_points(self):
        """Number of vertices and thus the length of the input to set_values."""
        return len(self._points)

    def set_lookup_table(self, lut):
        """Set the colorscale."""
        self._vtk_mapper.SetLookupTable(lut)
        self._vtk_mapper.Modified()

    def get_coordinates(self):
        """Get the coordinates for which values should be provided. These correspond to
        the vertices of the polygons making up the sphere."""
        return self._points

    def get_actor(self):
        """Return the vtk actor responsible for the sphere."""
        return self._vtk_actor

    def _get_inner_index(self, index_0, index_1):
        """The displacement of index of a vertex in a triangle compared to the base corner. """
        return int((self._sampling_n-2)*(index_0-1) + ((index_0-1)/2.) - ((index_0-1)**2/2.) + (index_1-1))

    def _generate_points_and_polys(self):
        """Calculate the coordinate positions with the current n and group them into polygons."""
        import itertools
        from eke import icosahedral_sphere

        coordinates = icosahedral_sphere.icosahedron_vertices()
        full_coordinate_list = [c for c in coordinates]

        edges = []
        edge_indices = []
        for coordinate_1, coordinate_2 in itertools.combinations(enumerate(coordinates), 2):
            if ((coordinate_1[1] == coordinate_2[1]).sum() < 3) and (_numpy.linalg.norm(coordinate_1[1] - coordinate_2[1]) < 3.):
                edges.append((coordinate_1[1], coordinate_2[1]))
                edge_indices.append((coordinate_1[0], coordinate_2[0]))

        edge_table = {}
        for edge in edge_indices:
            edge_table[edge] = []
            edge_table[edge[::-1]] = edge_table[edge]

        face_cutoff = 1.5
        faces = []
        face_indices = []
        for coordinate_1, coordinate_2, coordinate_3 in itertools.combinations(enumerate(coordinates), 3):
            if  (((coordinate_1[1] == coordinate_2[1]).sum() < 3) and
                 ((coordinate_1[1] == coordinate_3[1]).sum() < 3) and
                 ((coordinate_2[1] == coordinate_3[1]).sum() < 3)):
                center = (coordinate_1[1]+coordinate_2[1]+coordinate_3[1])/3.
                if  (_numpy.linalg.norm(center-coordinate_1[1]) < face_cutoff and
                     _numpy.linalg.norm(center-coordinate_2[1]) < face_cutoff and
                     _numpy.linalg.norm(center-coordinate_3[1]) < face_cutoff):
                    faces.append((coordinate_1[1], coordinate_2[1], coordinate_3[1]))
                    face_indices.append((coordinate_1[0], coordinate_2[0], coordinate_3[0]))

        for edge_index, edge in zip(edge_indices, edges):
            diff = edge[1] - edge[0]
            edge_table[edge_index].append(edge_index[0])
            for edge_position in _numpy.linspace(0., 1., self._sampling_n+1)[1:-1]:
                full_coordinate_list.append(edge[0] + diff*edge_position)
                edge_table[edge_index].append(len(full_coordinate_list)-1)
            edge_table[edge_index].append(edge_index[1])

        polygons = []

        for face_index in face_indices:
            edge_0 = edge_table[(face_index[0], face_index[1])]
            edge_1 = edge_table[(face_index[0], face_index[2])]
            edge_2 = edge_table[(face_index[1], face_index[2])]

            current_face_offset = len(full_coordinate_list)
            origin = full_coordinate_list[edge_0[0]]
            base_0 = full_coordinate_list[edge_0[-1]] - origin
            base_1 = full_coordinate_list[edge_1[-1]] - origin

            base_1o = _numpy.cross(origin, base_0)
            dot_product = _numpy.dot(base_1o, base_1)
            if dot_product < 0.:
                tmp = base_0
                base_0 = base_1
                base_1 = tmp
                tmp = edge_0
                edge_0 = edge_1
                edge_1 = tmp
                edge_2 = edge_2[::-1]

            for index_0 in range(1, self._sampling_n):
                for index_1 in range(1, self._sampling_n):
                    if index_0+index_1 < self._sampling_n:
                        full_coordinate_list.append(origin + index_0/float(self._sampling_n)*base_0 + index_1/float(self._sampling_n)*base_1)

            # no inner points
            polygons.append((edge_0[0], edge_0[1], edge_1[1]))
            polygons.append((edge_0[-2], edge_0[-1], edge_2[1]))
            polygons.append((edge_1[-2], edge_2[-2], edge_1[-1]))

            # one inner points
            # corner faces
            polygons.append((edge_0[1], current_face_offset+self._get_inner_index(1, 1), edge_1[1]))
            polygons.append((edge_0[-2], edge_2[1], current_face_offset+self._get_inner_index(self._sampling_n-2, 1)))
            polygons.append((edge_1[-2], current_face_offset + self._get_inner_index(1, self._sampling_n-2), edge_2[-2]))

            # edge faces
            for index in range(1, self._sampling_n-1):
                polygons.append((edge_0[index], edge_0[index+1], current_face_offset+self._get_inner_index(index, 1)))
                polygons.append((edge_1[index], current_face_offset+self._get_inner_index(1, index), edge_1[index+1]))
                polygons.append((edge_2[index], edge_2[index+1], current_face_offset+self._get_inner_index(self._sampling_n-1-index, index)))

            # two inner points
            for index in range(2, self._sampling_n-1):
                polygons.append((edge_0[index], current_face_offset+self._get_inner_index(index, 1),
                                 current_face_offset+self._get_inner_index(index-1, 1)))
                polygons.append((edge_1[index], current_face_offset+self._get_inner_index(1, index-1),
                                 current_face_offset+self._get_inner_index(1, index)))
                polygons.append((edge_2[index], current_face_offset+self._get_inner_index(self._sampling_n-1-index, index),
                                 current_face_offset+self._get_inner_index(self._sampling_n-index, index-1)))

            # three inner points
            for index_0 in range(1, self._sampling_n-2):
                for index_1 in range(1, self._sampling_n-2):
                    if index_0 + index_1 < self._sampling_n-1:
                        polygons.append((current_face_offset+self._get_inner_index(index_0, index_1),
                                         current_face_offset+self._get_inner_index(index_0+1, index_1),
                                         current_face_offset+self._get_inner_index(index_0, index_1+1)))
            for index_0 in range(2, self._sampling_n-2):
                for index_1 in range(1, self._sampling_n-3):
                    if index_0 + index_1 < self._sampling_n-1:
                        polygons.append((current_face_offset+self._get_inner_index(index_0, index_1),
                                         current_face_offset+self._get_inner_index(index_0, index_1+1),
                                         current_face_offset+self._get_inner_index(index_0-1, index_1+1)))

        for point in full_coordinate_list:
            point /= _numpy.linalg.norm(point)

        self._points = _numpy.array(full_coordinate_list)
        self._polygons = _numpy.array(polygons)

    def _generate_vtk_objects(self):
        """Generate vtk_points, vtk_polygons, vtk_poly_data, vtk_scalars,
        vtk_mapper and vtk_actor.
        This function must be called after _generate_points_and_polys()"""
        self._vtk_points = _vtk.vtkPoints()
        for coordinates in self._points:
            self._vtk_points.InsertNextPoint(coordinates[0], coordinates[1], coordinates[2])

        self._vtk_polygons = _vtk.vtkCellArray()
        for polygon in self._polygons:
            vtk_polygon = _vtk.vtkPolygon()
            vtk_polygon.GetPointIds().SetNumberOfIds(3)
            for local_index, global_index in enumerate(polygon):
                vtk_polygon.GetPointIds().SetId(local_index, global_index)
            self._vtk_polygons.InsertNextCell(vtk_polygon)

        self._vtk_poly_data = _vtk.vtkPolyData()
        self._vtk_poly_data.SetPoints(self._vtk_points)
        self._vtk_poly_data.SetPolys(self._vtk_polygons)

        self._vtk_scalars = _vtk.vtkFloatArray()
        self._vtk_scalars.SetNumberOfValues(self._vtk_poly_data.GetPoints().GetNumberOfPoints())
        for i in range(self._vtk_scalars.GetNumberOfTuples()):
            self._vtk_scalars.SetValue(i, 0.)

        self._vtk_poly_data.GetPointData().SetScalars(self._vtk_scalars)
        self._vtk_poly_data.Modified()

        self._vtk_mapper = _vtk.vtkPolyDataMapper()
        if VTK_VERSION < 6:
            self._vtk_mapper.SetInput(self._vtk_poly_data)
        else:
            self._vtk_mapper.SetInputData(self._vtk_poly_data)
        self._vtk_mapper.InterpolateScalarsBeforeMappingOn()
        self._vtk_mapper.UseLookupTableScalarRangeOn()
        self._vtk_actor = _vtk.vtkActor()
        self._vtk_actor.SetMapper(self._vtk_mapper)

        normals = _vtk.vtkFloatArray()
        normals.SetNumberOfComponents(3)
        for point in self._points:
            normals.InsertNextTuple(point)
        self._vtk_poly_data.GetPointData().SetNormals(normals)


    def _update_vtk_objects(self):
        """When n is changed the thus the number of coordinates this function is needed
        to update the vtk objects with the new number of points."""
        # self._vtk_points.SetNumberOfPoints(len(self._points))
        # for i, c in enumerate(self._points):
        #     self._vtk_points.InsertPoint(i, c[0], c[1], c[2])
        self._vtk_points = _vtk.vtkPoints()
        for coordinates in self._points:
            self._vtk_points.InsertNextPoint(coordinates[0], coordinates[1], coordinates[2])

        self._vtk_polygons = _vtk.vtkCellArray()
        for polygon in self._polygons:
            vtk_polygon = _vtk.vtkPolygon()
            vtk_polygon.GetPointIds().SetNumberOfIds(3)
            for local_index, global_index in enumerate(polygon):
                vtk_polygon.GetPointIds().SetId(local_index, global_index)
            self._vtk_polygons.InsertNextCell(vtk_polygon)

        self._vtk_poly_data.SetPoints(self._vtk_points)
        self._vtk_poly_data.SetPolys(self._vtk_polygons)

        self._vtk_scalars = _vtk.vtkFloatArray()
        self._vtk_scalars.SetNumberOfValues(self._vtk_poly_data.GetPoints().GetNumberOfPoints())
        for i in range(self._vtk_scalars.GetNumberOfTuples()):
            self._vtk_scalars.SetValue(i, 0.)

        self._vtk_poly_data.GetPointData().SetScalars(self._vtk_scalars)
        self._vtk_poly_data.Modified()

class IsoSurface(object):
    """Generate and plot isosurfacs."""
    def __init__(self, volume, level=None, spacing=(1., 1., 1.)):
        self._surface_algorithm = None
        self._renderer = None
        self._actor = None
        self._mapper = None
        self._volume_array = None

        self._float_array = _vtk.vtkFloatArray()
        self._image_data = _vtk.vtkImageData()
        self._image_data.GetPointData().SetScalars(self._float_array)
        self._setup_data(_numpy.float32(volume))
        self._image_data.SetSpacing(spacing[2], spacing[1], spacing[0])

        self._surface_algorithm = _vtk.vtkMarchingCubes()
        if VTK_VERSION >= 6:
            self._surface_algorithm.SetInputData(self._image_data)
        else:
            self._surface_algorithm.SetInput(self._image_data)
        self._surface_algorithm.ComputeNormalsOn()

        if level is not None:
            try:
                self.set_multiple_levels(iter(level))
            except TypeError:
                self.set_level(0, level)

        self._mapper = _vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._surface_algorithm.GetOutputPort())
        self._mapper.ScalarVisibilityOn() # new
        self._actor = _vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

    def _setup_data(self, volume):
        """Setup the relation between the numpy volume, the vtkFloatArray and vtkImageData."""
        self._volume_array = _numpy.zeros(volume.shape, dtype="float32", order="C")
        self._volume_array[:] = volume
        self._float_array.SetNumberOfValues(_numpy.product(volume.shape))
        self._float_array.SetNumberOfComponents(1)
        self._float_array.SetVoidArray(self._volume_array, _numpy.product(volume.shape), 1)
        self._image_data.SetDimensions(*(self._volume_array.shape[::-1]))

    def set_renderer(self, renderer):
        """Set the vtkRenderer to render the isosurfaces. Adding a new renderer will remove the last one."""
        if self._actor is None:
            raise RuntimeError("Actor does not exist.")
        if self._renderer is not None:
            self._renderer.RemoveActor(self._actor)
        self._renderer = renderer
        self._renderer.AddActor(self._actor)

    def set_multiple_levels(self, levels):
        """Remova any current surface levels and add the ones from the provided list."""
        self._surface_algorithm.SetNumberOfContours(0)
        for index, this_level in enumerate(levels):
            self._surface_algorithm.SetValue(index, this_level)
        self._render()

    def get_levels(self):
        """Return a list of the current surface levels."""
        return [self._surface_algorithm.GetValue(index) for index in range(self._surface_algorithm.GetNumberOfContours())]

    def add_level(self, level):
        """Add a single surface level."""
        self._surface_algorithm.SetValue(self._surface_algorithm.GetNumberOfContours(), level)
        self._render()

    def remove_level(self, index):
        """Remove a singel surface level at the provided index."""
        for index in range(index, self._surface_algorithm.GetNumberOfContours()-1):
            self._surface_algorithm.SetValue(index, self._surface_algorithm.GetValue(index+1))
        self._surface_algorithm.SetNumberOfContours(self._surface_algorithm.GetNumberOfContours()-1)
        self._render()

    def set_level(self, index, level):
        """Change the value of an existing surface level."""
        self._surface_algorithm.SetValue(index, level)
        self._render()

    def set_cmap(self, cmap):
        """Set the colormap. Supports all matplotlib colormaps."""
        self._mapper.ScalarVisibilityOn()
        self._mapper.SetLookupTable(get_lookup_table(self._volume_array.min(), self._volume_array.max(), colorscale=cmap))
        self._render()

    def set_color(self, color):
        """Plot all surfaces in the provided color. Accepts an rbg iterable."""
        self._mapper.ScalarVisibilityOff()
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._render()

    def set_opacity(self, opacity):
        """Set the opacity of all surfaces. (seting it individually for each surface is not supported)"""
        self._actor.GetProperty().SetOpacity(opacity)
        self._render()

    def _render(self):
        """Render if a renderer is set, otherwise do nothing."""
        if self._renderer is not None:
            self._renderer.GetRenderWindow().Render()

    def set_data(self, volume):
        """Change the data displayed. The new array must have the same shape as the current one."""
        if volume.shape != self._volume_array.shape:
            raise ValueError("New volume must be the same shape as the old one")
        self._volume_array[:] = volume
        self._float_array.Modified()
        self._render()

def plot_isosurface(volume, level=None, spacing=(1., 1., 1.)):
    """Plot isosurfaces of the provided module. Levels can be iterable or singel value."""
    surface_object = IsoSurface(volume, level, spacing)

    renderer = _vtk.vtkRenderer()
    render_window = _vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = _vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(_vtk.vtkInteractorStyleRubberBandPick())

    #renderer.AddActor(surface_object.get_actor())
    surface_object.set_renderer(renderer)

    renderer.SetBackground(0., 0., 0.)
    render_window.SetSize(800, 800)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

class InteractiveIsosurface(QtWidgets.QMainWindow):
    def __init__(self, volume, spacing=(1., 1., 1.)):
        super(InteractiveIsosurface, self).__init__()
        self._default_size = (600, 600)
        self.resize(*self._default_size)

        #self._volume = numpy.ascontiguousarray(volume, dtype="float32")
        self._surface_object = IsoSurface(volume, spacing)

        self._central_widget = QtWidgets.QWidget(self)
        self._vtk_widget = QVTKRenderWindowInteractor(self._central_widget)
        self._vtk_widget.SetInteractorStyle(_vtk.vtkInteractorStyleRubberBandPick())

        self._renderer = _vtk.vtkRenderer()
        self._renderer.SetBackground(0., 0., 0.)

        self._surface_object.set_renderer(self._renderer)

        self._THRESHOLD_SLIDER_MAXIMUM = 1000
        self._THRESHOLD_SLIDER_INIT = self._THRESHOLD_SLIDER_MAXIMUM/2
        self._threshold_table = self._adaptive_slider_values(volume, self._THRESHOLD_SLIDER_MAXIMUM, volume.min(), volume.max())
        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._threshold_slider.setMaximum(self._THRESHOLD_SLIDER_MAXIMUM)

        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self._vtk_widget)
        self._layout.addWidget(self._threshold_slider)

        self._central_widget.setLayout(self._layout)
        self.setCentralWidget(self._central_widget)

    def initialize(self):
        self._vtk_widget.Initialize()
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
        self._threshold_slider.valueChanged.connect(self._threshold_slider_changed)
        self._threshold_slider.setValue(self._THRESHOLD_SLIDER_INIT)

    def _threshold_slider_changed(self, value):
        surface_level = self._threshold_table[value]
        self._surface_object.set_level(0, surface_level)

    @staticmethod
    def _adaptive_slider_values(volume, slider_maximum, vmin, vmax):
        unique_values = _numpy.unique(_numpy.sort(volume.flat))
        unique_values = unique_values[(unique_values >= vmin) * (unique_values <= vmax)]
        interpolator = _scipy.interpolate.interp1d(_numpy.arange(len(unique_values)), unique_values)
        level_table = interpolator(_numpy.linspace(0., len(unique_values)-1, slider_maximum+1))
        return level_table

def plot_isosurface_interactive(volume, spacing=(1., 1., 1)):
    #app = QtGui.QApplication(["Interactive IsoSurface"])
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["Interactive IsoSurface"])
    interactive_isosurface = InteractiveIsosurface(volume, spacing)
    interactive_isosurface.show()
    interactive_isosurface.initialize()
    interactive_isosurface.activateWindow()
    interactive_isosurface.show()
    interactive_isosurface.raise_()
    app.exec_()
    #print 1./0.
    #app.processEvents()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["Foo"])


def plot_planes(array_in, spacing=(1., 1., 1.), log=False, cmap=None):
    """Plot two interactive planes cutting the provided volume."""
    array_in = _numpy.float64(array_in)
    renderer = _vtk.vtkRenderer()
    render_window = _vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = _vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(_vtk.vtkInteractorStyleRubberBandPick())

    if cmap == None:
        import matplotlib as _matplotlib
        cmap = _matplotlib.rcParams["image.cmap"]
    lut = get_lookup_table(max(0., array_in.min()), array_in.max(), log=log, colorscale=cmap)
    picker = _vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    image_data = array_to_image_data(array_in)
    image_data.SetSpacing(spacing[2], spacing[1], spacing[0])

    def setup_plane():
        """Create and setup a singel plane."""
        plane = _vtk.vtkImagePlaneWidget()
        if VTK_VERSION >= 6:
            plane.SetInputData(image_data)
        else:
            plane.SetInput(image_data)
        plane.UserControlledLookupTableOn()
        plane.SetLookupTable(lut)
        plane.DisplayTextOn()
        plane.SetPicker(picker)
        plane.SetLeftButtonAction(1)
        plane.SetMiddleButtonAction(2)
        plane.SetRightButtonAction(0)
        plane.SetInteractor(interactor)
        return plane

    plane_1 = setup_plane()
    plane_1.SetPlaneOrientationToXAxes()
    plane_1.SetSliceIndex(int(array_in.shape[0]*spacing[0]/2))
    plane_1.SetEnabled(1)
    plane_2 = setup_plane()
    plane_2.SetPlaneOrientationToYAxes()
    plane_2.SetSliceIndex(int(array_in.shape[1]*spacing[1]/2))
    plane_2.SetEnabled(1)

    renderer.SetBackground(0., 0., 0.)
    render_window.SetSize(800, 800)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

class SynchronizedInteractorStyle(_vtk.vtkInteractorStyleRubberBandPick):
    def __init__(self):
        #super(SynchronizedInteractorStyle, self).__init__()
        self._moving = False
        self._panning = False
        self._shift_pressed = False
        self._renderers = []

        self.AddObserver("MouseMoveEvent", self._mouse_move_event)
        self.AddObserver("LeftButtonPressEvent", self._left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self._left_button_release_event)
        self.AddObserver("MiddleButtonPressEvent", self._middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self._middle_button_release_event)
        self.AddObserver("RightButtonPressEvent", self._right_button_press_event)
        self.AddObserver("RightButtonReleaseEvent", self._right_button_release_event)
        self.AddObserver("MouseWheelForwardEvent", self._mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent", self._mouse_wheel_backward_event)
        self.AddObserver("KeyPressEvent", self._key_press_event)
        self.AddObserver("KeyReleaseEvent", self._key_release_event)
        # self.AddObserver("KeyDownEvent", self._key_down_event)
        # self.AddObserver("KeyUpEvent", self._key_up_event)

    def _render_all(self):
        for renderer in self._renderers:
            renderer.GetRenderWindow().Render()

    def _left_button_press_event(self, obj, event):
        # if self.GetShiftKey():
        #     self._panning = True
        # else:
        #     self._moving = True
        self._moving = True
        self.OnLeftButtonDown()

    def _left_button_release_event(self, obj, event):
        self._moving = False
        self._panning = False
        self.OnLeftButtonDown()

    def _middle_button_press_event(self, obj, event):
        self._moving = True
        self.OnMiddleButtonDown()

    def _middle_button_release_event(self, obj, event):
        self._moving = False
        self.OnMiddleButtonDown()

    def _right_button_press_event(self, obj, event):
        self._moving = True
        self.OnRightButtonDown()

    def _right_button_release_event(self, obj, event):
        self._moving = False
        self.OnRightButtonDown()

    def _mouse_wheel_forward_event(self, obj, event):
        self.OnMouseWheelForward()
        self._render_all()

    def _mouse_wheel_backward_event(self, obj, event):
        self.OnMouseWheelBackward()
        self._render_all()

    def _mouse_move_event(self, obj, event):
        if self._moving:
            if self._shift_pressed:
                self.Pan()
            else:
                self.Rotate()
            self._render_all()

    def _key_press_event(self, obj, event):
        if self.GetInteractor().GetShiftKey():
            self._shift_pressed = True

    def _key_release_event(self, obj, event):
        if not self.GetInteractor().GetShiftKey():
            self._shift_pressed = False

    def call(self, function):
        def new_function(obj, event):
            self._render_all()
            function()
        return new_function

    def add_renderer(self, renderer):
        self._renderers.append(renderer)

def synchronize_renderers(renderer_list):
    for renderer in renderer_list:
        render_window = renderer.GetRenderWindow()
        interactor = render_window.GetInteractor()
        my_interactor_style = SynchronizedInteractorStyle()
        interactor.SetInteractorStyle(my_interactor_style)
        my_interactor_style.add_renderer(renderer)
        for other_renderer in renderer_list:
            if other_renderer is not renderer:
                my_interactor_style.add_renderer(other_renderer)
    camera = renderer_list[0].GetActiveCamera()
    for renderer in renderer_list:
        renderer.SetActiveCamera(camera)

def setup_window(size=(800, 800), background=(1., 1., 1.)):
    """Returns (renderer, render_window, interactor)"""
    renderer = _vtk.vtkRenderer()
    #renderer.SetUseDepthPeeling(True)
    render_window = _vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = _vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(_vtk.vtkInteractorStyleRubberBandPick())
    interactor.SetRenderWindow(render_window)

    renderer.SetBackground(background[0], background[1], background[2])
    render_window.SetSize(size[0], size[1])

    interactor.Initialize()
    render_window.Render()
    return renderer, render_window, interactor



class ScatterPlot(object):
    """Class for plotting multiple dataset in one scatterplot."""
    def __init__(self):
        self._renderer, self._render_window, self._interactor = setup_window()

        self._bounds = _numpy.array((0., 1., 0., 1., 0., 1.))
        self._axes_actor = _vtk.vtkCubeAxesActor()
        self._axes_actor.SetBounds(self._bounds)

        self._axes_actor.SetCamera(self._renderer.GetActiveCamera())
        self._axes_actor.SetFlyModeToStaticTriad()

        self._axes_actor.GetXAxesLinesProperty().SetColor(0., 0., 0.)
        self._axes_actor.GetYAxesLinesProperty().SetColor(0., 0., 0.)
        self._axes_actor.GetZAxesLinesProperty().SetColor(0., 0., 0.)
        for i in range(3):
            self._axes_actor.GetLabelTextProperty(i).SetColor(0., 0., 0.)
            self._axes_actor.GetTitleTextProperty(i).SetColor(0., 0., 0.)

        self._renderer.AddActor(self._axes_actor)

    def plot(self, data, color="black", color_value=None, cmap="viridis", point_size=None, pixel_point=False):
        """Add a 3D scatterplot to the current plot. Data is given as an array
        of size (N, 3). Color denotes the one color used for every
        point while color_value is an array of scalars that are used
        to color the points with a given color map.

        """

        if len(data.shape) != 2 or data.shape[1] != 3:
            raise ValueError("data must have shape (n, 3) where n is the number of points.")

        data = _numpy.float32(data)
        data_vtk = array_to_float_array(data)
        point_data = _vtk.vtkPoints()
        point_data.SetData(data_vtk)
        points_poly_data = _vtk.vtkPolyData()
        points_poly_data.SetPoints(point_data)

        if not color_value is None:
            lut = get_lookup_table(color_value.min(), color_value.max(), colorscale=cmap)
            color_scalars = array_to_vtk(_numpy.float32(color_value.copy()))
            color_scalars.SetLookupTable(lut)
            points_poly_data.GetPointData().SetScalars(color_scalars)

        import matplotlib
        color_rgb = matplotlib.colors.to_rgb(color)


        if pixel_point:
            if point_size is None:
                point_size = 3
            glyph_filter = _vtk.vtkVertexGlyphFilter()
            glyph_filter.SetInputData(points_poly_data)
            glyph_filter.Update()
        else:
            if point_size is None:
                point_size = _numpy.array(data).std() / len(data)**(1./3.) / 3.
            glyph_filter = _vtk.vtkGlyph3D()
            glyph_filter.SetInputData(points_poly_data)
            sphere_source = _vtk.vtkSphereSource()
            sphere_source.SetRadius(point_size)
            glyph_filter.SetSourceConnection(sphere_source.GetOutputPort())
            glyph_filter.SetScaleModeToDataScalingOff()
            if not color_value is None:
                glyph_filter.SetColorModeToColorByScalar()
            else:
                glyph_filter.SetColorMode(0)
            glyph_filter.Update()

        poly_data = _vtk.vtkPolyData()
        poly_data.ShallowCopy(glyph_filter.GetOutput())

        mapper = _vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        if color_value is not None:
            mapper.SetLookupTable(lut)
            mapper.SetUseLookupTableScalarRange(True)

        points_actor = _vtk.vtkActor()
        points_actor.SetMapper(mapper)
        points_actor.GetProperty().SetPointSize(point_size)
        points_actor.GetProperty().SetColor(*color_rgb)

        new_bounds = _numpy.array(points_actor.GetBounds())
        self._bounds[0::2] = _numpy.minimum(self._bounds[0::2], new_bounds[0::2])
        self._bounds[1::2] = _numpy.maximum(self._bounds[1::2], new_bounds[1::2])
        self._axes_actor.SetBounds(self._bounds)

        self._renderer.AddActor(points_actor)

        self._render_window.Render()

    def start(self):
        self._interactor.Start()

def scatterplot(data, color="black", color_value=None, cmap="viridis", point_size=None, pixel_point=False):
    """Plot a 3D scatterplot. Data is given as an array of size (N,
    3). Color denotes the one color used for every point while
    color_value is an array of scalars that are used to color the
    points with a given color map.

    """
    plot_object = ScatterPlot()
    plot_object.plot(data, color, color_value, cmap, point_size, pixel_point)
    plot_object.start()
