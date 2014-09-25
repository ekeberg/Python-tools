import vtk
import numpy

def get_lookup_table(minimum_value, maximum_value, log=False, colorscale="jet", number_of_colors=1000):
    import matplotlib
    import matplotlib.cm
    if log:
        lut = vtk.vtkLogLookupTable()
    else:
        lut = vtk.vtkLookupTable()
    lut.SetTableRange(minimum_value, maximum_value)
    lut.SetNumberOfColors(number_of_colors)
    lut.Build()
    for i in range(number_of_colors):
        color = matplotlib.cm.cmap_d[colorscale](float(i) / float(number_of_colors))
        lut.SetTableValue(i, color[0], color[1], color[2], 1.)
    return lut


class SphereMap(object):
    def __init__(self, n):
        self._n = n
        self._generate_points_and_polys()
        self._generate_vtk_objects()

    def set_n(self, n):
        self._n = n
        self._generate_points_and_polys()
        self._update_vtk_objects()

    def set_values(self, values):
        if len(values) != self._vtk_scalars.GetNumberOfTuples():
            raise ValueError("Input value array must match number of points")
        for i in range(self._vtk_scalars.GetNumberOfTuples()):
            self._vtk_scalars.SetValue(i, values[i])
        self._vtk_scalars.Modified()

    def get_number_of_points(self):
        return len(self._points)

    def set_lookup_table(self, lut):
        self._vtk_mapper.SetLookupTable(lut)
        self._vtk_mapper.Modified()

    def get_coordinates(self):
        return self._points

    def get_actor(self):
        return self._vtk_actor

    def _get_inner_index(self, index_0, index_1):
        return int((self._n-2)*(index_0-1) + ((index_0-1)/2.) - ((index_0-1)**2/2.) + (index_1-1))

    def _generate_points_and_polys(self):
        import itertools
        import icosahedral_sphere

        coordinates = icosahedral_sphere.icosahedron_vertices()
        full_coordinate_list = [c for c in coordinates]

        edges = []
        edge_indices = []
        for c1, c2 in itertools.combinations(enumerate(coordinates), 2):
            if ((c1[1] == c2[1]).sum() < 3) and (numpy.linalg.norm(c1[1] - c2[1]) < 3.):
                edges.append((c1[1], c2[1]))
                edge_indices.append((c1[0], c2[0]))

        edge_table = {}
        for edge in edge_indices:
            edge_table[edge] = []
            edge_table[edge[::-1]] = edge_table[edge]

        face_cutoff = 1.5
        faces = []
        face_indices = []
        for c1, c2, c3 in itertools.combinations(enumerate(coordinates), 3):
            if ((c1[1]==c2[1]).sum() < 3) and ((c1[1]==c3[1]).sum() < 3) and ((c2[1]==c3[1]).sum() < 3):
                center = (c1[1]+c2[1]+c3[1])/3.
                if (numpy.linalg.norm(center-c1[1]) < face_cutoff and
                    numpy.linalg.norm(center-c2[1]) < face_cutoff and
                    numpy.linalg.norm(center-c3[1]) < face_cutoff):
                    faces.append((c1[1], c2[1], c3[1]))
                    face_indices.append((c1[0], c2[0], c3[0]))

        for edge_index, edge in zip(edge_indices, edges):
            diff = edge[1] - edge[0]
            edge_table[edge_index].append(edge_index[0])
            for s in numpy.linspace(0., 1., self._n+1)[1:-1]:
                full_coordinate_list.append(edge[0] + diff*s)
                edge_table[edge_index].append(len(full_coordinate_list)-1)
            edge_table[edge_index].append(edge_index[1])

        polygons = []

        for face_index, face in zip(face_indices, faces):
            edge_0 = edge_table[(face_index[0], face_index[1])]
            edge_1 = edge_table[(face_index[0], face_index[2])]
            edge_2 = edge_table[(face_index[1], face_index[2])]

            current_face_offset = len(full_coordinate_list)
            origin = full_coordinate_list[edge_0[0]]
            base_0 = full_coordinate_list[edge_0[-1]] - origin
            base_1 = full_coordinate_list[edge_1[-1]] - origin

            for index_0 in range(1, self._n):
                for index_1 in range(1, self._n):
                    if index_0+index_1 < self._n:
                        full_coordinate_list.append(origin + index_0/float(self._n)*base_0 + index_1/float(self._n)*base_1)

            # no inner points
            polygons.append((edge_0[0], edge_0[1], edge_1[1]))
            polygons.append((edge_0[-2], edge_0[-1], edge_2[1]))
            polygons.append((edge_1[-2], edge_2[-2], edge_1[-1]))

            # one inner points
            # corner faces
            polygons.append((edge_0[1], current_face_offset+self._get_inner_index(1, 1), edge_1[1]))
            polygons.append((edge_0[-2], edge_2[1], current_face_offset+self._get_inner_index(self._n-2, 1)))
            polygons.append((edge_1[-2], current_face_offset + self._get_inner_index(1, self._n-2), edge_2[-2]))

            # edge faces
            for index in range(1, self._n-1):
                polygons.append((edge_0[index], edge_0[index+1], current_face_offset+self._get_inner_index(index, 1)))
                polygons.append((edge_1[index], edge_1[index+1], current_face_offset+self._get_inner_index(1, index)))
                polygons.append((edge_2[index], edge_2[index+1], current_face_offset+self._get_inner_index(self._n-1-index, index)))

            # two inner points
            for index in range(2, self._n-1):
                polygons.append((edge_0[index], current_face_offset+self._get_inner_index(index, 1),
                                 current_face_offset+self._get_inner_index(index-1, 1)))
                polygons.append((edge_1[index], current_face_offset+self._get_inner_index(1, index-1),
                                 current_face_offset+self._get_inner_index(1, index)))
                polygons.append((edge_2[index], current_face_offset+self._get_inner_index(self._n-1-index, index),
                                 current_face_offset+self._get_inner_index(self._n-index, index-1)))

            # three inner points
            for index_0 in range(1, self._n-2):
                for index_1 in range(1, self._n-2):
                    if index_0 + index_1 < self._n-1:
                        polygons.append((current_face_offset+self._get_inner_index(index_0, index_1),
                                         current_face_offset+self._get_inner_index(index_0+1, index_1),
                                         current_face_offset+self._get_inner_index(index_0, index_1+1)))
            for index_0 in range(2, self._n-2):
                for index_1 in range(1, self._n-3):
                    if index_0 + index_1 < self._n-1:
                        polygons.append((current_face_offset+self._get_inner_index(index_0, index_1),
                                         current_face_offset+self._get_inner_index(index_0, index_1+1),
                                         current_face_offset+self._get_inner_index(index_0-1, index_1+1)))

        for p in full_coordinate_list:
            p /= numpy.linalg.norm(p)

        self._points = numpy.array(full_coordinate_list)
        self._polygons = numpy.array(polygons)

    def _generate_vtk_objects(self):
        """Generate vtk_points, vtk_polygons, vtk_poly_data, vtk_scalars,
        vtk_mapper and vtk_actor.
        This function must be called after _generate_points_and_polys()"""
        self._vtk_points = vtk.vtkPoints()
        for c in self._points:
            self._vtk_points.InsertNextPoint(c[0], c[1], c[2])

        self._vtk_polygons = vtk.vtkCellArray()
        for p in self._polygons:
            vtk_polygon = vtk.vtkPolygon()
            vtk_polygon.GetPointIds().SetNumberOfIds(3)
            for i, pi in enumerate(p):
                vtk_polygon.GetPointIds().SetId(i, pi)
            self._vtk_polygons.InsertNextCell(vtk_polygon)

        self._vtk_poly_data = vtk.vtkPolyData()
        self._vtk_poly_data.SetPoints(self._vtk_points)
        self._vtk_poly_data.SetPolys(self._vtk_polygons)

        self._vtk_scalars = vtk.vtkFloatArray()
        self._vtk_scalars.SetNumberOfValues(self._vtk_poly_data.GetPoints().GetNumberOfPoints())
        for i in range(self._vtk_scalars.GetNumberOfTuples()):
            self._vtk_scalars.SetValue(i, 0.)

        self._vtk_poly_data.GetPointData().SetScalars(self._vtk_scalars)
        self._vtk_poly_data.Modified()

        self._vtk_mapper = vtk.vtkPolyDataMapper()
        self._vtk_mapper.SetInputData(self._vtk_poly_data)
        self._vtk_mapper.InterpolateScalarsBeforeMappingOn()
        self._vtk_mapper.UseLookupTableScalarRangeOn()
        self._vtk_actor = vtk.vtkActor()
        self._vtk_actor.SetMapper(self._vtk_mapper)

    def _update_vtk_objects(self):
        # self._vtk_points.SetNumberOfPoints(len(self._points))
        # for i, c in enumerate(self._points):
        #     self._vtk_points.InsertPoint(i, c[0], c[1], c[2])
        self._vtk_points = vtk.vtkPoints()
        for c in self._points:
            self._vtk_points.InsertNextPoint(c[0], c[1], c[2])

        self._vtk_polygons = vtk.vtkCellArray()
        for p in self._polygons:
            vtk_polygon = vtk.vtkPolygon()
            vtk_polygon.GetPointIds().SetNumberOfIds(3)
            for i, pi in enumerate(p):
                vtk_polygon.GetPointIds().SetId(i, pi)
            self._vtk_polygons.InsertNextCell(vtk_polygon)

        self._vtk_poly_data.SetPoints(self._vtk_points)
        self._vtk_poly_data.SetPolys(self._vtk_polygons)

        self._vtk_scalars = vtk.vtkFloatArray()
        self._vtk_scalars.SetNumberOfValues(self._vtk_poly_data.GetPoints().GetNumberOfPoints())
        for i in range(self._vtk_scalars.GetNumberOfTuples()):
            self._vtk_scalars.SetValue(i, 0.)

        self._vtk_poly_data.GetPointData().SetScalars(self._vtk_scalars)
        self._vtk_poly_data.Modified()
        

