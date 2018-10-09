"""Plot slices of fourier space for illustratory purposes."""
import numpy as _numpy
import vtk as _vtk
import nfft as _nfft
from . import vtk_tools
from . import rotations
from . import diffraction

def downsample_pattern(image, factor):
    """Image shape must be exact multiples of downsample factor"""
    raise NotImplementedError()
    from scipy import ndimage
    size_y, size_x = image.shape
    y_2d, x_2d = _numpy.ogrid[:size_y, :size_x]
    regions = size_x//factor * (y_2d//factor) + x_2d/factor
    result = ndimage.mean(image, labels=regions,
                          index=_numpy.arange(regions.max()+1))
    result.shape = (size_y//factor, size_x/factor)


class Generate(object):
    """Generate slices of Fourier space. image_shape is in pixels and
    curvature is also given in pixels. Cutoff determines what part of
    Fourier space that will be used. Should be in the range of (0. 0.5]."""
    def __init__(self, real_volume, real_pixel_size, image_shape, wavelength, detector_distance, detector_pixel_size, cutoff=0.5):
        self._real_volume = real_volume
        self._image_shape = image_shape
        self._real_pixel_size = float(real_pixel_size)
        self._wavelength = float(wavelength)
        self._detector_distance = float(detector_distance)
        self._detector_pixel_size = float(detector_pixel_size)
        self._coordinates = diffraction.ewald_coordinates((self._image_shape, )*2, self._wavelength,
                                                          self._detector_distance, self._detector_pixel_size)

    def _rotated_coordinates(self, rot):
        """Return base coordinates rotated by rot. Rot is a quaternion."""
        rotated_coordinates = rotations.rotate_array(rot, self._coordinates)
        z_rotated = rotated_coordinates[:, 0].reshape(self._image_shape)
        y_rotated = rotated_coordinates[:, 1].reshape(self._image_shape)
        x_rotated = rotated_coordinates[:, 2].reshape(self._image_shape)

        coordinates = _numpy.transpose(_numpy.array((z_rotated.flatten(),
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

        pattern_flat = _nfft.nfft(self._real_volume, self._real_pixel_size, coordinates)

        pattern = pattern_flat.reshape(self._image_shape)
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
    def __init__(self, shape, wavelength, detector_distance, detector_pixel_size, slice_shape="square"):
        self._shape = shape
        # self._curvature = curvature
        self._wavelength = float(wavelength)
        self._detector_distance = float(detector_distance)
        self._detector_pixel_size = float(detector_pixel_size)

        self._coordinates = diffraction.ewald_coordinates(self._shape, self._wavelength, self._detector_distance,
                                                          self._detector_pixel_size).reshape(self._shape + (3, ))
        self._coordinates *= 1./abs(self._coordinates).max()
        
        
        self._image_values = _vtk.vtkFloatArray()
        self._image_values.SetNumberOfComponents(1)
        self._image_values.SetName("Intensity")

        self._points = _vtk.vtkPoints()
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                self._points.InsertNextPoint(self._coordinates[i, j, 0],
                                             self._coordinates[i, j, 1],
                                             self._coordinates[i, j, 2])
                self._image_values.InsertNextTuple1(0.)

        self._polygons = _vtk.vtkCellArray()

        if slice_shape is "square":
            self._square_slice()
        elif slice_shape is "circle":
            self._circular_slice()
        else:
            raise ValueError("Unknown slice shape: {}".format(slice_shape))
        self._template_poly_data = _vtk.vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def _square_slice(self):
        """Call in the beginning. Precalculates the polydata object
        without rotation."""
        self._polygons.Initialize()
        for i in range(self._shape[0]-1):
            for j in range(self._shape[1]-1):
                corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                polygon = _vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(4)
                for index, corner in enumerate(corners):
                    polygon.GetPointIds().SetId(index,
                                                corner[0]*self._shape[1]+corner[1])
                self._polygons.InsertNextCell(polygon)

        self._template_poly_data = _vtk.vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def _circular_slice(self):
        """Call in the beginning. Precalculates the polydata object
        without rotation."""
        radius = min(self._shape)/2.-1.5
        self._polygons.Initialize()
        for i in range(self._shape[0]-1):
            for j in range(self._shape[1]-1):
                if (i-self._shape[0]/2 + 0.5)**2 + (j-self._shape[1]/2 + 0.5)**2 < radius**2:
                    corners = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
                    polygon = _vtk.vtkPolygon()
                    polygon.GetPointIds().SetNumberOfIds(4)
                    for index, corner in enumerate(corners):
                        polygon.GetPointIds().SetId(index,
                                                    corner[0]*self._shape[1]+corner[1])
                    self._polygons.InsertNextCell(polygon)

        self._template_poly_data = _vtk.vtkPolyData()
        self._template_poly_data.SetPoints(self._points)
        self._template_poly_data.GetPointData().SetScalars(self._image_values)
        self._template_poly_data.SetPolys(self._polygons)

    def insert_slice(self, image, rotation):
        """Return a vtkPolyData object corresponding to a single
        slice with intensities given by the image variable and
        with the specified rotation. Rotation is a quaternion."""
        rotation_degrees = rotation.copy()
        rotation_degrees[0] = 2.*_numpy.arccos(rotation[0])*180./_numpy.pi
        transformation = _vtk.vtkTransform()
        transformation.RotateWXYZ(rotation_degrees[0], rotation_degrees[1],
                                  rotation_degrees[2], rotation_degrees[3])
        input_poly_data = _vtk.vtkPolyData()
        input_poly_data.DeepCopy(self._template_poly_data)
        transform_filter = _vtk.vtkTransformFilter()
        transform_filter.SetInputData(input_poly_data)
        transform_filter.SetTransform(transformation)
        transform_filter.Update()
        this_poly_data = transform_filter.GetOutput()

        scalars = this_poly_data.GetPointData().GetScalars()
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                scalars.SetTuple1(i*self._shape[1]+j, image[i, j])
        this_poly_data.Modified()
        return this_poly_data


class Plot(object):
    """Create vtk actors of curved slices and add them to the
    given vtkRenderer."""
    def __init__(self, renderer, image_shape, wavelength, detector_distance, detector_pixel_size, cmap="jet", slice_shape="square"):
        self._renderer = renderer
        self._generator = SliceGenerator(image_shape, wavelength, detector_distance, detector_pixel_size, slice_shape)
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

        mapper = _vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.UseLookupTableScalarRangeOn()
        mapper.SetLookupTable(self._lut)
        mapper.Modified()
        mapper.Update()
        actor = _vtk.vtkActor()
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

    def set_lut(self, lut):
        """NOT IMPLEMENTED. Provide a user-defined colorscale
        as a vtkLookupTable"""
        self._lut = lut
