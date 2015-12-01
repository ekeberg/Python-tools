"""Functions and classes that are useful when working with VTK. Uses vtk 6."""
import vtk as _vtk
import numpy as _numpy

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
    return lut

def vtk_from_array(array_in, dtype=None):
    """Get vtkFloatArray/vtkDoubleArray from the input array."""
    if dtype == None:
        dtype = array_in.dtype
    if dtype == "float32":
        float_array = _vtk.vtkFloatArray()
    elif dtype == "float64":
        float_array = _vtk.vtkDoubleArray()
    else:
        raise ValueError("Wrong format of input array, must be float32 or float64")
    float_array.SetNumberOfComponents(1)
    float_array.SetVoidArray(_numpy.ascontiguousarray(array_in, dtype))
    return float_array

def window_to_png(render_window, file_name, magnification=1):
    window_to_image_filter = _vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetMagnification(magnification)
    window_to_image_filter.SetInputBufferTypeToRGBA()
    window_to_image_filter.Update()

    writer = _vtk.vtkPNGWriter()
    writer.SetFileName(file_name)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()


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
        import icosahedral_sphere

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
    """!!Ths function is not done!! Plot an isosurface."""
    def __init__(self, volume):
        self._surface_level = _numpy.mean((volume.min(), volume.max()))
        self._actor = None
        self._volume_scalars = None
        self._image_shape = None
        self._volume_max = None
        self._volume_array = None
        self._algorithm = None
        self._mapper = None
        self._volume = None

        self._color = (0., 1., 0.)
        self._set_volume(volume)

    def _set_volume(self, volume):
        """Set a 3D numpy array of the density that should be plotted without
        updating any of the vtk objects."""
        self._image_shape = volume.shape
        self._volume_array = volume

    def set_volume(self, volume):
        """Set a 3D numpy array of the density that should be plotted."""
        self._set_volume(volume)
        #what else

    def get_actor(self):
        """Return the vtk actor responsible for the isosurface."""
        return self._actor

    def get_data(self):
        """Return the volume array for easy replacement elsewhere."""
        pass

    def set_color(self, color):
        """Change the color of the surface. Takes indexable of length 3."""
        if len(color) != 3:
            raise ValueError("Color must have length 3. (Was {0})".format(len(color)))
        self._color = color

    def _generate_vtk_volume(self):
        """Generate the vtkImageData object and connect it to a numpy array."""
        self._volume_max = 1.

        self._volume = _vtk.vtkImageData()
        self._volume.SetDimensions(*self._image_shape)

        self._volume_scalars = _vtk.vtkFloatArray()
        self._volume_scalars.SetNumberOfValues(_numpy.product(self._image_shape))
        self._volume_scalars.SetNumberOfComponents(1)
        self._volume_scalars.SetName("FooName")

        for i in range(_numpy.product(self._image_shape)):
            self._volume_scalars.SetValue(i, self._volume_array)

        self._volume.GetPointData().SetScalars(self._volume_scalars)

    def _setup_surface(self):
        """create the isosurface object and plotting objects."""
        self._algorithm = _vtk.vtkMarchingCubes()
        self._algorithm.SetInputData(self._volume)
        self._algorithm.ComputeNormalsOn()
        self._algorithm.SetValue(0, self._surface_level)

        self._mapper = _vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._algorithm.GetOutputPort())
        self._mapper.ScalarVisibilityOff()
        self._actor = _vtk.vtkActor()
        self._actor.GetProperty().SetColor(*self._color)
        self._actor.SetMapper(self._mapper)


