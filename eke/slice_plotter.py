"""Plot slices of fourier space for illustratory purposes."""
import numpy
import vtk
import rotations
import nfft
import vtk_tools


def downsample_pattern(image, factor):
    """Image shape must be exact multiples of downsample factor"""
    from scipy import ndimage
    size_y, size_x = image.shape
    y_2d, x_2d = numpy.ogrid[:size_y, :size_x]
    regions = size_x/factor * (y_2d/factor) + x_2d/factor
    result = ndimage.mean(image, labels=regions,
                          index=numpy.arange(regions.max()+1))
    result.shape = (size_y/factor, size_x/factor)


class Generate(object):
    """Generate slices of Fourier space. image_side is in pixels and
    curvature is also given in pixels."""
    def __init__(self, real_volume, image_side, curvature):
        self._real_volume = real_volume
        self._image_side = image_side
        self._pixel_size_fourier = 1./image_side
        self._curvature = curvature*self._pixel_size_fourier
        self._x_base_2d = None
        self._y_base_2d = None
        self._z_base_2d = None
        self._calculate_base_coordinates()

    def _calculate_base_coordinates(self):
        """Base coordinates are in x-y plane and are scaled so that the
        range in x and y is from (-0.5,0.5)"""
        x_base_coordinates = (self._pixel_size_fourier *
                              numpy.linspace(-self._image_side/2+0.5,
                                             self._image_side/2-0.5,
                                             self._image_side))
        y_base_coordinates = (self._pixel_size_fourier *
                              numpy.linspace(-self._image_side/2+0.5,
                                             self._image_side/2-0.5,
                                             self._image_side))
        self._y_base_2d, self._x_base_2d = numpy.meshgrid(y_base_coordinates,
                                                          x_base_coordinates)
        self._z_base_2d = (self._curvature -
                           numpy.sqrt(self._curvature**2 - self._x_base_2d**2 -
                                      self._y_base_2d**2))

    def _rotated_coordinates(self, rot):
        """Return base coordinates rotated by rot. Rot is a quaternion."""
        (z_rotated,
         y_rotated,
         x_rotated) = rotations.rotate_array(rot, self._z_base_2d.flatten(),
                                             self._y_base_2d.flatten(),
                                             self._x_base_2d.flatten())
        z_rotated = z_rotated.reshape((self._image_side, )*2)
        y_rotated = y_rotated.reshape((self._image_side, )*2)
        x_rotated = x_rotated.reshape((self._image_side, )*2)

        coordinates = numpy.transpose(numpy.array((z_rotated.flatten(),
                                                   y_rotated.flatten(),
                                                   x_rotated.flatten())))
        return coordinates

    def get_slice_and_rot(self, rot=None, output_type="complex"):
        """Calculate the slice of the specified rot or if no rot is given
        use a random rotation. Both the slice and the rotation in returned.
        The output type can be either "complex" or "intensity". "Complex"
        returns the wave field and "intensity" returns the absolute square
        of the wave field."""
        if rot is None:
            rot = rotations.random_quaternion()
        coordinates = self._rotated_coordinates(rot)

        pattern_flat = nfft.nfft(self._real_volume, coordinates)

        pattern = pattern_flat.reshape((self._image_side, )*2)
        if output_type == "complex":
            return pattern, rot
        elif output_type == "intensity":
            return abs(pattern**2), rot

    def get_slice(self, rot=None, output_type="complex"):
        """Calculate the slice of the specified rot or if no rot is given
        use a random rotation. Returns the slice only.
        The output type can be either "complex" or "intensity". "Complex"
        returns the wave field and "intensity" returns the absolute square
        of the wave field."""
        pattern, _ = self.get_slice_and_rot(rot, output_type)
        return pattern


class SliceGenerator(object):
    """Generate multiple vtkPolyData objects corresponding to a curved
    slice throug Fourier space."""
    def __init__(self, side, curvature):
        self._side = side
        self._curvature = curvature
        x_array_single = numpy.arange(self._side) - self._side/2. + 0.5
        y_array_single = numpy.arange(self._side) - self._side/2. + 0.5
        y_array, x_array = numpy.meshgrid(y_array_single, x_array_single)
        z_array = (self._curvature - numpy.sqrt(self._curvature**2 -
                                                x_array**2 - y_array**2))

        self._image_values = vtk.vtkFloatArray()
        self._image_values.SetNumberOfComponents(1)
        self._image_values.SetName("Intensity")

        self._points = vtk.vtkPoints()
        for i in range(self._side):
            for j in range(self._side):
                self._points.InsertNextPoint(x_array[i, j],
                                             y_array[i, j],
                                             z_array[i, j])
                self._image_values.InsertNextTuple1(0.)

        self._polygons = vtk.vtkCellArray()

        self._square_slice()
        self._template_poly_data = vtk.vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def _square_slice(self):
        """Call in the beginning. Precalculates the polydata object
        without rotation."""
        self._polygons.Initialize()
        for i in range(self._side-1):
            for j in range(self._side-1):
                corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(4)
                for index, corner in enumerate(corners):
                    polygon.GetPointIds().SetId(index,
                                                corner[0]*self._side+corner[1])
                self._polygons.InsertNextCell(polygon)

        self._template_poly_data = vtk.vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def insert_slice(self, image, rotation):
        """Return a vtkPolyData object corresponding to a single
        slice with intensities given by the image variable and
        with the specified rotation. Rotation is a quaternion."""
        rotation_degrees = rotation.copy()
        rotation_degrees[0] = 2.*numpy.arccos(rotation[0])*180./numpy.pi
        transformation = vtk.vtkTransform()
        transformation.RotateWXYZ(-rotation_degrees[0], rotation_degrees[3],
                                  rotation_degrees[2], rotation_degrees[1])
        input_poly_data = vtk.vtkPolyData()
        input_poly_data.DeepCopy(self._template_poly_data)
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(input_poly_data)
        transform_filter.SetTransform(transformation)
        transform_filter.Update()
        this_poly_data = transform_filter.GetOutput()

        scalars = this_poly_data.GetPointData().GetScalars()
        for i in range(self._side):
            for j in range(self._side):
                scalars.SetTuple1(i*self._side+j, image[i, j])
        this_poly_data.Modified()
        return this_poly_data


class Plot(object):
    """Create vtk actors of curved slices and add them to the
    given vtkRenderer."""
    def __init__(self, renderer, image_side, curvature, cmap="jet"):
        self._renderer = renderer
        self._generator = SliceGenerator(image_side, curvature)
        self._lut = vtk_tools.get_lookup_table(1., 1., log=True,
                                               colorscale=cmap)
        self._custom_lut_range = {"min": False, "max": False}

    def insert_slice(self, image, rotation):
        """Create and add a slice with the specified intensities
        and rotation. rotation is a quaternion."""
        poly_data = self._generator.insert_slice(image, rotation)

        lut_changed = False
        if (not self._custom_lut_range["max"] and
                image.max() > self._lut.GetRange()[1]):
            self._lut.SetRange(self._lut.GetRange()[0], image.max())
            lut_changed = True
        if (not self._custom_lut_range["min"] and
                image.min() < self._lut.GetRange()[0]):
            self._lut.SetRange(image.min(), self._lut.GetRange()[1])
            lut_changed = True
        if lut_changed:
            self._lut.Build()
            self._lut.Modified()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.UseLookupTableScalarRangeOn()
        mapper.SetLookupTable(self._lut)
        mapper.Modified()
        mapper.Update()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self._renderer.AddViewProp(actor)

    def set_cmap_range(self, cmin=None, cmax=None):
        """Set maximum and/or minimum value of the colormap. The
        specified variable will no longer be updated automatically."""
        if cmin is not None:
            self._custom_lut_range["min"] = True
            self._lut.SetRange(self._lut.GetRange()[0], cmin)
        if cmax is not None:
            self._custom_lut_range["max"] = True
            self._lut.SetRange(cmax, self._lut.GetRange()[1])

    def set_lut(self):
        """NOT IMPLEMENTED. Provide a user-defined colorscale
        as a vtkLookupTable"""
        pass
