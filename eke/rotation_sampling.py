
import numpy as _numpy
import itertools as _itertools


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

    angle_max = 0.5
    angle_step = 0.05

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

def _base_vertices():
    from eke import rotmodule
    import itertools
    import scipy.constants

    rotations = _numpy.zeros((120, 4))

    # first 16
    rotations[:16, :] = _numpy.array(list(itertools.product(*((-0.5, 0.5), )*4)))

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
    for v1, v2, v3 in itertools.product(*base[:3]):
        b = (v1, v2, v3, 0)
        for p in permutations:
            rotations[24+counter] = (b[p[0]], b[p[1]], b[p[2]], b[p[3]])
            counter += 1
    return rotations

def _edge_indices():
    verts = _base_vertices()
    products = _numpy.linalg.norm(verts[:, _numpy.newaxis, :] + verts[_numpy.newaxis, :, :], axis=2)

    products *= ~_numpy.triu(_numpy.ones((120, )*2, dtype="bool8"))
    mask = products > 1.8

    i1, i2 = _numpy.meshgrid(range(120), range(120), indexing="ij")
    all_i1 = i1[mask]
    all_i2 = i2[mask]
    return all_i1, all_i2
    

def _face_indices():
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
    return all_i1, all_i2, all_i3
            

def _cell_indices():
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
    return all_i1, all_i2, all_i3, all_i4
    

# def rotation_sampling(n):
#     from eke import rotmodule
#     import itertools
#     import scipy.constants

#     rotations = _numpy.zeros((2*rotmodule.n_to_rots(n), 4))

#     # first 16
#     rotations[:16, :] = _numpy.array(list(itertools.product(*((-0.5, 0.5), )*4)))

#     # next 8
#     rotations[16:20] = _numpy.identity(4)
#     rotations[20:24] = -_numpy.identity(4)

#     # next 96
#     permutations = ((0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1),
#                     (1, 2, 0, 3), (1, 3, 2, 0), (1, 0, 3, 2),
#                     (2, 0, 1, 3), (2, 3, 0, 1), (2, 1, 3, 0),
#                     (3, 1, 0, 2), (3, 2, 1, 0), (3, 0, 2, 1))
    
#     base = ((-0.5, 0.5),
#             (-0.5*scipy.constants.golden_ratio, 0.5*scipy.constants.golden_ratio),
#             (-0.5/scipy.constants.golden_ratio, 0.5/scipy.constants.golden_ratio),
#             0)

#     counter = 0
#     for v1, v2, v3 in itertools.product(*base[:3]):
#         b = (v1, v2, v3, 0)
#         for p in permutations:
#             print(counter)
#             rotations[24+counter] = (b[p[0]], b[p[1]], b[p[2]], b[p[3]])
#             counter += 1

#     # edges
#     # edge_start_ind, endge_end_ind = _edge_indices()
#     # for start_ind, end_ind in zip(edge_start_ind, edge_end_ind):
#     #     steps = linspace(0, 1, n+1)[1:-1]
#     #     steps[:, _numpy.newaxis]*

#     return rotations

# def rotation_sampling(n):
#     verts = _base_vertices()

#     edges = _edge_indices()
#     faces = _face_indices()
#     cells = _cell_indices()

#     split_cells = 
    
