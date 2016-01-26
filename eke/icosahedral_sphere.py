"""Module to create point coordinates that sample the surface of a sphere close to uniformly.
Also contains functions to generate face and edge coordinates of an icosahedron."""
import numpy as _numpy
import itertools as _itertools

_PHI = (1+_numpy.sqrt(5))/2.

def n_to_points(sampling_n):
    """How many sampling points corresponds to a specific n."""
    if sampling_n <= 0:
        raise ValueError("sampling_n must be positive.")
    return 2+10*sampling_n**2

def points_to_n(n_points):
    sampling_n = (n_points-2) / 10.
    if sampling_n != int(sampling_n):
        print sampling_n
        raise ValueError("{0} points does not correspond to any n".format(n_points))
    return int(sampling_n)

def icosahedron_vertices():
    """Retern the coordinates of the 12 vertices of an icosahedron
    with edge length 2."""
    coordinates = []
    coordinates.append(_numpy.array((0., +1, +_PHI)))
    coordinates.append(_numpy.array((0., -1, +_PHI)))
    coordinates.append(_numpy.array((0., +1, -_PHI)))
    coordinates.append(_numpy.array((0., -1, -_PHI)))

    coordinates.append(_numpy.array((+1, +_PHI, 0.)))
    coordinates.append(_numpy.array((-1, +_PHI, 0.)))
    coordinates.append(_numpy.array((+1, -_PHI, 0.)))
    coordinates.append(_numpy.array((-1, -_PHI, 0.)))

    coordinates.append(_numpy.array((+_PHI, 0., +1)))
    coordinates.append(_numpy.array((+_PHI, 0., -1)))
    coordinates.append(_numpy.array((-_PHI, 0., +1)))
    coordinates.append(_numpy.array((-_PHI, 0., -1)))

    #coordinates = _numpy.array(coordinates)
    return coordinates

def icosahedron_edges():
    """Return a list of all the 30 edges in an icosahedron. Each edge is a two-length tuple
    with the coordinates of the start and end point."""
    coordinates = icosahedron_vertices()
    edges = []
    # for c1 in coordinates:
    #     for c2 in coordinates:
    for vertex_1, vertex_2 in _itertools.combinations(coordinates, 2):
        if ((vertex_1 == vertex_2).sum() < 3) and (_numpy.linalg.norm(vertex_1 - vertex_2) < 3.):
            edges.append((vertex_1, vertex_2))
    return edges

def icosahedron_faces():
    """Return a list of all the 20 faces in an icosahedron. Each face is a three-length tuple
    with the coordinates of the three vertices."""
    coordinates = icosahedron_vertices()
    face_cutoff = 1.5
    faces = []
    for vertex_1, vertex_2, vertex_3 in _itertools.combinations(coordinates, 3):
        if  ((vertex_1 == vertex_2).sum() < 3 and
             (vertex_1 == vertex_3).sum() < 3 and
             (vertex_2 == vertex_3).sum() < 3):
            center = (vertex_1+vertex_2+vertex_3)/3.
            if  (_numpy.linalg.norm(center-vertex_1) < face_cutoff and
                 _numpy.linalg.norm(center-vertex_2) < face_cutoff and
                 _numpy.linalg.norm(center-vertex_3) < face_cutoff):
                faces.append((vertex_1, vertex_2, vertex_3))
    return faces

def sphere_sampling(n):
    """Return a sampling on the sphere based on an icosahedron with a fine
    sampling of the faces. n determines the number of subdivisions of each
    triangle."""
    coordinates = icosahedron_vertices()
    edges = icosahedron_edges()
    faces = icosahedron_faces()

    edge_points = []

    for edge in edges:
        origin = edge[0]
        base = edge[1]-edge[0]
        for i in range(1, n):
            edge_points.append(origin + i/float(n)*base)

    face_points = []
    for face in faces:
        origin = face[0]
        base_1 = face[1]-face[0]
        base_2 = face[2]-face[0]

        # Swap vertex order to make sure it is always counter clockwise
        base_2o = _numpy.cross(origin, base_1)
        dot_product = _numpy.dot(base_2o, base_2)
        if dot_product < 0.:
            tmp = base_1
            base_1 = base_2
            base_2 = tmp

        for i in range(1, n):
            for j in range(1, n):
                if i+j < n:
                    point = origin + i/float(n)*base_1 + j/float(n)*base_2
                    face_points.append(point)

    full_list = [_numpy.array(c) for c in coordinates] + edge_points + face_points
    normalized_list = [l/_numpy.linalg.norm(l) for l in full_list]
    return normalized_list

