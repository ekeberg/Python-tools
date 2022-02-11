
import numpy as _numpy
import itertools as _itertools
import pathlib as _pathlib
import h5py as _h5py

DATA_FILE = _pathlib.Path(__file__).parents[0] / "data/600_cell.h5"

def _create_and_save_data():
    print("600-cell data does not exist. Generating. This can take a few minutes.")
    DATA_FILE.parents[0].mkdir(exist_ok=True)
    with _h5py.File(DATA_FILE, "w") as file_handle:
        file_handle.create_dataset("vertices", data=_base_vertices(allow_cached=False))
        file_handle.create_dataset("edge_indices", data=_edge_indices(allow_cached=False))
        file_handle.create_dataset("face_indices", data=_face_indices(allow_cached=False))
        file_handle.create_dataset("cell_indices", data=_cell_indices(allow_cached=False))

def local_rotations(angle_max, angle_step):
    if angle_max > 0.65:
        raise Warning("local_rotation doesn't support angles higher than 0.65")
        
    golden_ratio = (1. + _numpy.sqrt(5))/2.

    vertices = []

    vertices += _itertools.product((-0.5, 0.5),
                                  (-0.5, 0.5),
                                  (-0.5, 0.5),
                                  (-0.5, 0.5))


    vertices += [_numpy.roll((1., 0., 0., 0.), i) for i in range(4)]
    vertices += [_numpy.roll((-1., 0., 0., 0.), i) for i in range(4)]

    # all even permutations
    permutations = _numpy.array([(0,1,2,3),
                                (0,3,1,2),
                                (0,2,3,1),
                                (1,2,0,3),
                                (1,3,2,0),
                                (1,0,3,2),
                                (2,0,1,3),
                                (2,3,0,1),
                                (2,1,3,0),
                                (3,1,0,2),
                                (3,2,1,0),
                                (3,0,2,1)])

    for s1, s2, s3 in _itertools.product((-1, 1), (-1, 1), (-1, 1)):
        vertices += [(0.5*_numpy.array([s1*golden_ratio, s2*1., s3*1./golden_ratio, 0.]))[this_permutation] for this_permutation in permutations]

    vertices = _numpy.array(vertices)

    cutoff = 1./golden_ratio + 0.001

    region = vertices[_numpy.linalg.norm(vertices - _numpy.array((1., 0., 0., 0.)), axis=1) < cutoff]

    origin = region[0]
    directions = (_numpy.array(region[1:]) - origin)*golden_ratio

    cutoff_radius = _numpy.sqrt(1. - _numpy.cos(angle_max/2.)**2)
    step_size = _numpy.sqrt(1. - _numpy.cos(angle_step/2.)**2)

    # Identify neighbour pairs

    all_pairs = _numpy.array(list(_itertools.combinations(directions, 2)))
    pairs = all_pairs[_numpy.array([_numpy.linalg.norm(p1 - p2) for p1, p2 in all_pairs]) < 1.01]

    # Identify triplets

    all_triplets = _numpy.array(list(_itertools.combinations(directions, 3)))
    tripplets = all_triplets[_numpy.array([_numpy.linalg.norm(p1 + p2 + p3) for p1, p2, p3 in all_triplets]) > 2.4]

    # Lines
    lines = []
    for this_direction in directions:
        lines.append(origin + _numpy.arange(step_size, cutoff_radius, step_size)[:, _numpy.newaxis] * this_direction[_numpy.newaxis, :])

    # Planes
    planes = []
    for direction0, direction1 in pairs:
        this_plane = (_numpy.arange(step_size, cutoff_radius, step_size)[:, _numpy.newaxis, _numpy.newaxis] * direction0 +
                      _numpy.arange(step_size, cutoff_radius, step_size)[_numpy.newaxis, :, _numpy.newaxis] * direction1)
        this_plane = this_plane.reshape((this_plane.shape[0]*this_plane.shape[1], this_plane.shape[2]))
        this_plane_pruned = this_plane[_numpy.linalg.norm(this_plane, axis=1) < cutoff_radius]
        planes.append(origin + this_plane_pruned)

    # Volumes

    volumes = []
    for direction0, direction1, direction2 in tripplets:
        this_volume = (_numpy.arange(step_size, cutoff_radius, step_size)[:, _numpy.newaxis, _numpy.newaxis, _numpy.newaxis] * direction0 +
                       _numpy.arange(step_size, cutoff_radius, step_size)[_numpy.newaxis, :, _numpy.newaxis, _numpy.newaxis] * direction1 +
                       _numpy.arange(step_size, cutoff_radius, step_size)[_numpy.newaxis, _numpy.newaxis, :, _numpy.newaxis] * direction2)
        this_volume = this_volume.reshape((_numpy.prod(this_volume.shape[:-1]), this_volume.shape[-1]))
        this_volume_pruned = this_volume[_numpy.linalg.norm(this_volume, axis=1) < cutoff_radius]
        volumes.append(origin + this_volume_pruned)

    local_rots = _numpy.vstack((_numpy.array([origin]),
                               _numpy.concatenate(lines),
                               _numpy.concatenate(planes),
                               _numpy.concatenate(volumes)))
    local_rots = _numpy.array(local_rots)

    local_rots /= _numpy.linalg.norm(local_rots, axis=1)[:, _numpy.newaxis]

    return local_rots


def _base_vertices_backend():
    import scipy.constants
    rotations = _numpy.zeros((120, 4))

    # first 16
    rotations[:16, :] = _numpy.array(list(_itertools.product(*((-0.5, 0.5), )*4)))

    # next 8
    rotations[16:20] = _numpy.identity(4)
    rotations[20:24] = -_numpy.identity(4)

    # next 96
    permutations = ((0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1),
                    (1, 2, 0, 3), (1, 3, 2, 0), (1, 0, 3, 2),
                    (2, 0, 1, 3), (2, 3, 0, 1), (2, 1, 3, 0),
                    (3, 1, 0, 2), (3, 2, 1, 0), (3, 0, 2, 1))
    
    base = ((-0.5, 0.5),
            (-0.5*scipy.constants.golden_ratio, 0.5*scipy.constants.golden_ratio),
            (-0.5/scipy.constants.golden_ratio, 0.5/scipy.constants.golden_ratio),
            0)

    counter = 0
    for v1, v2, v3 in _itertools.product(*base[:3]):
        b = (v1, v2, v3, 0)
        for p in permutations:
            rotations[24+counter] = (b[p[0]], b[p[1]], b[p[2]], b[p[3]])
            counter += 1
    return rotations

def _base_vertices(allow_cached=True):
    if not allow_cached:
        return _base_vertices_backend()
    elif not DATA_FILE.exists():
        _create_and_save_data()
    with _h5py.File(DATA_FILE, "r") as file_handle:
        return file_handle["vertices"][...]


def _edge_indices_backend():
    verts = _base_vertices()
    products = _numpy.linalg.norm(verts[:, _numpy.newaxis, :] + verts[_numpy.newaxis, :, :], axis=2)

    products *= ~_numpy.triu(_numpy.ones((120, )*2, dtype="bool8"))
    mask = products > 1.8

    i1, i2 = _numpy.meshgrid(range(120), range(120), indexing="ij")
    all_i1 = i1[mask]
    all_i2 = i2[mask]
    return _numpy.stack((all_i1, all_i2))

def _edge_indices(allow_cached=True):
    if not allow_cached:
        return _edge_indices_backend()
    elif not DATA_FILE.exists():
        _create_and_save_data()
    with _h5py.File(DATA_FILE, "r") as file_handle:
        return file_handle["edge_indices"][...]

def _face_indices_backend():
    verts = _base_vertices()
    products = _numpy.linalg.norm(verts[:, _numpy.newaxis, _numpy.newaxis, :] +
                                  verts[_numpy.newaxis, :, _numpy.newaxis, :] +
                                  verts[_numpy.newaxis, _numpy.newaxis, :, :], axis=3)

    mask_1d = _numpy.arange(120)
    mask = ((mask_1d[_numpy.newaxis, _numpy.newaxis, :] > mask_1d[_numpy.newaxis, :, _numpy.newaxis]) * 
            (mask_1d[_numpy.newaxis, :, _numpy.newaxis] > mask_1d[:, _numpy.newaxis, _numpy.newaxis]))
    products *= mask

    mask = products > 2.7

    i1, i2, i3 = _numpy.meshgrid(range(120), range(120), range(120), indexing="ij")
    all_i1 = i1[mask]
    all_i2 = i2[mask]
    all_i3 = i3[mask]
    return _numpy.stack((all_i1, all_i2, all_i3))

def _face_indices(allow_cached=True):
    if not allow_cached:
        return _face_indices_backend()
    elif not DATA_FILE.exists():
        _create_and_save_data()
    with _h5py.File(DATA_FILE, "r") as file_handle:
        return file_handle["face_indices"][...]


def _cell_indices_backend():
    verts = _base_vertices()
    products = _numpy.linalg.norm(verts[:, _numpy.newaxis, _numpy.newaxis, _numpy.newaxis, :] +
                                  verts[_numpy.newaxis, :, _numpy.newaxis, _numpy.newaxis, :] +
                                  verts[_numpy.newaxis, _numpy.newaxis, :, _numpy.newaxis, :] +
                                  verts[_numpy.newaxis, _numpy.newaxis, _numpy.newaxis, :, :], axis=4)
    mask_1d = _numpy.arange(120)
    mask = ((mask_1d[_numpy.newaxis, _numpy.newaxis, _numpy.newaxis, :] >
             mask_1d[_numpy.newaxis, _numpy.newaxis, :, _numpy.newaxis]) * 
            (mask_1d[_numpy.newaxis, _numpy.newaxis, :, _numpy.newaxis] >
             mask_1d[_numpy.newaxis, :, _numpy.newaxis, _numpy.newaxis]) *
            (mask_1d[_numpy.newaxis, :, _numpy.newaxis, _numpy.newaxis] >
             mask_1d[:, _numpy.newaxis, _numpy.newaxis, _numpy.newaxis]))
    products *= mask
    mask = products > 3.67
    # mask = products > 

    i1, i2, i3, i4 = _numpy.meshgrid(range(120), range(120), range(120), range(120), indexing="ij")
    all_i1 = i1[mask]
    all_i2 = i2[mask]
    all_i3 = i3[mask]
    all_i4 = i4[mask]
    return _numpy.stack((all_i1, all_i2, all_i3, all_i4))

def _cell_indices(allow_cached=True):
    if not allow_cached:
        return _cell_indices_backend()
    elif not DATA_FILE.exists():
        _create_and_save_data()
    with _h5py.File(DATA_FILE, "r") as file_handle:
        return file_handle["cell_indices"][...]
    

def _check_rot(rot):
    for r in rot:
        if r > 1e-10:
            return True
        if r < -1e-10:
            return False
    return False


def rotation_sampling(n):
    verts = _base_vertices()

    edges = _edge_indices()
    faces = _face_indices()
    cells = _cell_indices()

    all_points = []

    for v in verts:
        if _check_rot(v):
            all_points.append(v)
    
    for v1, v2 in zip(*edges):
        for i in range(1, n):
            new_point = (verts[v1] + (verts[v2]-verts[v1]) * (i/n))
            if _check_rot(new_point):
                all_points.append(new_point)

    for v1, v2, v3 in zip(*faces):
        for i1, i2 in _itertools.product(range(1, n), range(1, n)):
            if i1 + i2 < n:
                new_point = (verts[v1] +
                             (verts[v2] - verts[v1]) * (i1/n) +
                             (verts[v3] - verts[v1]) * (i2/n))
                if _check_rot(new_point):
                    all_points.append(new_point)
    
    for v1, v2, v3, v4 in zip(*cells):
        for i1, i2, i3 in _itertools.product(range(1, n), range(1, n), range(1, n)):
            if i1 + i2 + i3 < n:
                new_point = (verts[v1] +
                             (verts[v2]-verts[v1]) * (i1/n) +
                             (verts[v3]-verts[v1]) * (i2/n) +
                             (verts[v4]-verts[v1]) * (i3/n))
                if _check_rot(new_point):
                    all_points.append(new_point)

    all_points = _numpy.array(all_points)
    all_points /= _numpy.linalg.norm(all_points, axis=1)[:, _numpy.newaxis]
    return all_points
    
