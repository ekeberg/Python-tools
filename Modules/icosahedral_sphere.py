import pylab
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools

_phi = (1+pylab.sqrt(5))/2.

def icosahedron_vertices():
    coordinates = []
    coordinates.append(pylab.array((0., +1, +_phi)))
    coordinates.append(pylab.array((0., -1, +_phi)))
    coordinates.append(pylab.array((0., +1, -_phi)))
    coordinates.append(pylab.array((0., -1, -_phi)))

    coordinates.append(pylab.array((+1, +_phi, 0.)))
    coordinates.append(pylab.array((-1, +_phi, 0.)))
    coordinates.append(pylab.array((+1, -_phi, 0.)))
    coordinates.append(pylab.array((-1, -_phi, 0.)))

    coordinates.append(pylab.array((+_phi, 0., +1)))
    coordinates.append(pylab.array((+_phi, 0., -1)))
    coordinates.append(pylab.array((-_phi, 0., +1)))
    coordinates.append(pylab.array((-_phi, 0., -1)))

    #coordinates = pylab.array(coordinates)
    return coordinates

def icosahedron_edges():
    coordinates = icosahedron_vertices()
    edges = []
    # for c1 in coordinates:
    #     for c2 in coordinates:
    for c1, c2 in itertools.combinations(coordinates, 2):
        if ((c1 == c2).sum() < 3) and (pylab.norm(c1 - c2) < 3.):
            edges.append((c1, c2))
    return edges

def icosahedron_faces():
    coordinates = icosahedron_vertices()
    face_cutoff = 1.5
    faces = []
    for c1, c2, c3 in itertools.combinations(coordinates, 3):
        if ((c1==c2).sum() < 3) and ((c1==c3).sum() < 3) and ((c2==c3).sum() < 3):
            center = (c1+c2+c3)/3.
            if pylab.norm(center-c1) < face_cutoff and pylab.norm(center-c2) < face_cutoff and pylab.norm(center-c3) < face_cutoff:
                faces.append((c1, c2, c3))
    return faces

def sphere_sampling(n):
    coordinates = icosahedron_vertices()
    edges = icosahedron_edges()
    faces = icosahedron_faces()

    edge_points = []

    for e in edges:
        origin = e[0]
        base = e[1]-e[0]
        for i in range(1, n):
            edge_points.append(origin + i/float(n)*base)

    face_points = []
    for f in faces:
        origin = f[0]
        base_1 = f[1]-f[0]
        base_2 = f[2]-f[0]
        for i in range(1, n):
            for j in range(1, n):
                if i+j < n:
                    point = origin + i/float(n)*base_1 + j/float(n)*base_2
                    face_points.append(point)

    full_list = [pylab.array(c) for c in coordinates] + edge_points + face_points
    normalized_list =[l/pylab.norm(l) for l in full_list]
    return normalized_list

# points = sphere_sampling(5)

# class PointCompare(object):
#     def __init__(self, point_list):
#         self._point_list = point_list
        
#     def _get_closest(self, point):
        
#     def _compare(point0):
#         return pylab.dist(point, point0)
#     return min(list, key=point_compare)

