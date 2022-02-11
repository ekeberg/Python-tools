from numpy import float32
from functools import reduce
from operator import cmp


def invert_matrix(tf):
    from numpy import array, zeros, float
    tf = array(tf)
    r = tf[:, :3]
    t = tf[:, 3]
    tfinv = zeros((3, 4), float)
    rinv = tfinv[:, :3]
    tinv = tfinv[:, 3]
    from numpy.linalg import inv as matrix_inverse
    from numpy import dot as matrix_multiply
    rinv[:, :] = matrix_inverse(r)
    tinv[:] = matrix_multiply(rinv, -t)
    return tfinv


def skew_axes(cell_angles):
    if tuple(cell_angles) == (90, 90, 90):
        # Use exact trig functions for this common case.
        ca = cb = cg = c1 = 0
        sg = c2 = 1
    else:
        # Convert to radians
        from math import pi, sin, cos, sqrt
        alpha, beta, gamma = [a * pi / 180 for a in cell_angles]
        cg = cos(gamma)
        sg = sin(gamma)
        cb = cos(beta)
        ca = cos(alpha)
        c1 = (ca - cb*cg)/sg
        c2 = sqrt(1 - cb*cb - c1*c1)

    axes = ((1, 0, 0), (cg, sg, 0), (cb, c1, c2))
    return axes


# -----------------------------------------------------------------------------
# Maintain a cache of data objects using a limited amount of memory.
# The least recently accessed data is released first.
# -----------------------------------------------------------------------------
class Data_Cache:
    def __init__(self, size):
        self.size = size
        self.used = 0
        self.time = 1
        self.data = {}
        self.groups = {}

    def cache_data(self, key, value, size, description, groups=[]):
        self.remove_key(key)
        d = Cached_Data(key, value, size, description,
                        self.time_stamp(), groups)
        self.data[key] = d

        for g in groups:
            gtable = self.groups
        if g not in gtable:
            gtable[g] = []
        gtable[g].append(d)

        self.used = self.used + size
        self.reduce_use()

    def lookup_data(self, key):
        data = self.data
        if key in data:
            d = data[key]
            d.last_access = self.time_stamp()
            v = d.value
        else:
            v = None
        self.reduce_use()
        return v

    def remove_key(self, key):
        data = self.data
        if key in data:
            self.remove_data(data[key])
        self.reduce_use()

    def group_keys_and_data(self, group):
        groups = self.groups
        if group not in groups:
            return []

        kd = map(lambda d: (d.key, d.value), groups[group])
        return kd

    def resize(self, size):
        self.size = size
        self.reduce_use()

    def reduce_use(self):
        if self.used <= self.size:
            return

        data = self.data
        dlist = data.values()
        dlist.sort(lambda d1, d2: cmp(d1.last_access, d2.last_access))
        import sys
        for d in dlist:
            if sys.getrefcount(d.value) == 2:
                self.remove_data(d)
                if self.used <= self.size:
                    break

    def remove_data(self, d):
        del self.data[d.key]
        self.used = self.used - d.size
        d.value = None

        for g in d.groups:
            dlist = self.groups[g]
            dlist.remove(d)
            if len(dlist) == 0:
                del self.groups[g]

    def time_stamp(self):
        t = self.time
        self.time = t + 1
        return t


class Cached_Data:
    def __init__(self, key, value, size, description, time_stamp, groups):
        self.key = key
        self.value = value
        self.size = size
        self.description = description
        self.last_access = time_stamp
        self.groups = groups


data_cache = Data_Cache(size=0)


# -------------------------------------------- #
#         Grid_Data: datatype of chimera       #
# -------------------------------------------- #
class Grid_Data:
    def __init__(self, size,
                 value_type=float32,
                 origin=(0, 0, 0),
                 step=(1, 1, 1),
                 cell_angles=(90, 90, 90),
                 rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                 symmetries=(),
                 name='',
                 path='',       # Can be list of paths
                 file_type='',
                 grid_id='',
                 default_color=(.7, .7, .7, 1)):

        # Path, file_type and grid_id are for reloading data sets.
        self.path = path
        self.file_type = file_type  # 'mrc', 'spider', ....
        self.grid_id = grid_id  # String identifying grid in multi-grid files.

        if name == '':
            name = self.name_from_path(path)
        self.name = name

        self.size = tuple(size)

        from numpy import dtype
        if not isinstance(value_type, dtype):
            value_type = dtype(value_type)
        self.value_type = value_type        # numpy dtype.

        # Parameters defining how data matrix is positioned in space
        self.origin = tuple(origin)
        self.original_origin = self.origin
        self.step = tuple(step)
        self.original_step = self.step
        self.cell_angles = tuple(cell_angles)
        self.rotation = tuple(map(tuple, rotation))
        self.symmetries = symmetries
        self.ijk_to_xyz_transform = None
        self.xyz_to_ijk_transform = None

        self.rgba = default_color  # preferred color for displaying data

        global data_cache
        self.data_cache = data_cache

        self.writable = False
        self.change_callbacks = []

        self.update_transform()

    def set_path(self, path, format=None):
        if path != self.path:
            self.path = path
            self.name = self.name_from_path(path)
            self.call_callbacks('path changed')

        if format and format != self.file_type:
            self.file_type = format
            self.call_callbacks('file format changed')

    def name_from_path(self, path):
        from os.path import basename
        if isinstance(path, (list, tuple)):
            p = path[0]
        else:
            p = path
        name = basename(p)
        return name

    def set_origin(self, origin):
        if tuple(origin) != self.origin:
            self.origin = tuple(origin)
            self.update_transform()

    def set_step(self, step):
        if tuple(step) != self.step:
            self.step = tuple(step)
            self.update_transform()

    def set_cell_angles(self, cell_angles):
        if tuple(cell_angles) != self.cell_angles:
            self.cell_angles = tuple(cell_angles)
            self.update_transform()

    def set_rotation(self, rotation):
        r = tuple(map(tuple, rotation))
        if r != self.rotation:
            self.rotation = r
            self.update_transform()

    def update_transform(self):
        saxes = skew_axes(self.cell_angles)
        rsaxes = [apply_rotation(self.rotation, a) for a in saxes]
        tf, tf_inv = transformation_and_inverse(self.origin, self.step, rsaxes)
        if (
                tf != self.ijk_to_xyz_transform or
                tf_inv != self.xyz_to_ijk_transform
        ):
            self.ijk_to_xyz_transform = tf
            self.xyz_to_ijk_transform = tf_inv
            self.coordinates_changed()

    # ---------------------------------------------------------------------------
    # A matrix ijk corresponds to a point in xyz space.
    # This function maps the xyz point to the matrix index.
    # The returned matrix index need not be integers.
    #
    def xyz_to_ijk(self, xyz):
        return map_point(xyz, self.xyz_to_ijk_transform)

    # ---------------------------------------------------------------------------
    # A matrix ijk corresponds to a point in xyz space.
    # This function maps the matrix index to the xyz point.
    #
    def ijk_to_xyz(self, ijk):
        return map_point(ijk, self.ijk_to_xyz_transform)

    # ---------------------------------------------------------------------------
    # Spacings in xyz space of jk, ik, and ij planes.
    #
    def plane_spacings(self):
        spacings = map(lambda u: 1.0/norm(u[:3]), self.xyz_to_ijk_transform)
        return spacings

    # ---------------------------------------------------------------------------
    #
    def matrix(self, ijk_origin=(0, 0, 0), ijk_size=None,
               ijk_step=(1, 1, 1), progress=None, from_cache_only=False):
        if ijk_size is None:
            ijk_size = self.size

        m = self.cached_data(ijk_origin, ijk_size, ijk_step)
        if m is None and not from_cache_only:
            m = self.read_matrix(ijk_origin, ijk_size, ijk_step, progress)
            self.cache_data(m, ijk_origin, ijk_size, ijk_step)

        return m

    # ---------------------------------------------------------------------------
    # Must overide this function in derived class to return a 3 dimensional
    # NumPy matrix.  The returned matrix has size ijk_size and
    # element ijk is accessed as m[k,j,i].  It is an error if the requested
    # submatrix does not lie completely within the full data matrix.  It is
    # also an error for the size to be <= 0 in any dimension.  These invalid
    # inputs might throw an exception or might return garbage.  It is the
    # callers responsibility to make sure the arguments are valid.
    #
    def read_matrix(self, ijk_origin=(0, 0, 0), ijk_size=None,
                    ijk_step=(1, 1, 1), progress=None):
        raise NotImplementedError(f'Grid {self.name} has no read_matrix() '
                                  'routine')

    # ---------------------------------------------------------------------------
    # Convenience routine.
    #
    def matrix_slice(self, matrix, ijk_origin, ijk_size, ijk_step):
        i1, j1, k1 = ijk_origin
        i2, j2, k2 = map(lambda i, s: i+s, ijk_origin, ijk_size)
        istep, jstep, kstep = ijk_step
        m = matrix[k1:k2:kstep, j1:j2:jstep, i1:i2:istep]
        return m

    # ---------------------------------------------------------------------------
    # Deprecated.  Used before matrix() routine existed.
    #
    def full_matrix(self, progress=None):
        matrix = self.matrix()
        return matrix

    # ---------------------------------------------------------------------------
    # Deprecated.  Used before matrix() routine existed.
    #
    def submatrix(self, ijk_origin, ijk_size):
        return self.matrix(ijk_origin, ijk_size)

    # ---------------------------------------------------------------------------
    #
    def cached_data(self, origin, size, step):
        dcache = self.data_cache
        if dcache is None:
            return None

        key = (self, tuple(origin), tuple(size), tuple(step))
        m = dcache.lookup_data(key)
        if m is not None:
            return m

        # Look for a matrix containing the desired matrix
        group = self
        kd = dcache.group_keys_and_data(group)
        for k, matrix in kd:
            orig, sz, st = k[1:]
            if (
                    step[0] < st[0] or step[1] < st[1] or step[2] < st[2] or
                    step[0] % st[0] or step[1] % st[1] or step[2] % st[2]
            ):
                continue        # Step sizes not compatible
            if (
                    origin[0] < orig[0] or
                    origin[1] < orig[1] or
                    origin[2] < orig[2] or
                    origin[0] + size[0] > orig[0] + sz[0] or
                    origin[1] + size[1] > orig[1] + sz[1] or
                    origin[2] + size[2] > orig[2] + sz[2]
            ):
                continue        # Doesn't cover.
            dstep = map(lambda a, b: a / b, step, st)
            offset = map(lambda a, b: a - b, origin, orig)
            if offset[0] % st[0] or offset[1] % st[1] or offset[2] % st[2]:
                continue        # Offset stagger.
            moffset = map(lambda o, s: o / s, offset, st)
            msize = map(lambda s, t: (s+t-1) / t, size, st)
            m = matrix[moffset[2]:moffset[2]+msize[2]:dstep[2],
                       moffset[1]:moffset[1]+msize[1]:dstep[1],
                       moffset[0]:moffset[0]+msize[0]:dstep[0]]
            dcache.lookup_data(key)			# update access time
            return m

        return None

    # ---------------------------------------------------------------------------
    #
    def cache_data(self, m, origin, size, step):
        dcache = self.data_cache
        if dcache is None:
            return

        key = (self, tuple(origin), tuple(size), tuple(step))
        elements = reduce(lambda a, b: a * b, m.shape, 1)
        bytes = elements * m.itemsize
        groups = [self]
        descrip = self.data_description(origin, size, step)
        dcache.cache_data(key, m, bytes, descrip, groups)

    # ---------------------------------------------------------------------------
    #
    def data_description(self, origin, size, step):
        description = self.name

        if origin == (0, 0, 0):
            bounds = ' (%d,%d,%d)' % tuple(size)
        else:
            region = (origin[0], origin[0]+size[0]-1,
                      origin[1], origin[1]+size[1]-1,
                      origin[2], origin[2]+size[2]-1)
            bounds = ' (%d-%d,%d-%d,%d-%d)' % region
            description += bounds

        if step != (1, 1, 1):
            description += ' step (%d,%d,%d)' % tuple(step)

        return description

    # ---------------------------------------------------------------------------
    #
    def clear_cache(self):
        dcache = self.data_cache
        if dcache is None:
            return

        for k, d in dcache.group_keys_and_data(self):
            dcache.remove_key(k)

    # ---------------------------------------------------------------------------
    #
    def add_change_callback(self, cb):
        self.change_callbacks.append(cb)

    # ---------------------------------------------------------------------------
    #
    def remove_change_callback(self, cb):
        self.change_callbacks.remove(cb)

    # ---------------------------------------------------------------------------
    # Code has modified matrix elements, or the value type has changed.
    #
    def values_changed(self):
        self.call_callbacks('values changed')

    # ---------------------------------------------------------------------------
    # Mapping of array indices to xyz coordinates has changed.
    #
    def coordinates_changed(self):
        self.call_callbacks('coordinates changed')

    # ---------------------------------------------------------------------------
    #
    def call_callbacks(self, reason):
        for cb in self.change_callbacks:
            cb(reason)


# -----------------------------------------------------------------------------
# Return 3 by 4 matrix where first 3 columns give rotation and last column
# is translation.
#
def transformation_and_inverse(origin, step, axes):
    ox, oy, oz = origin
    d0, d1, d2 = step
    ax, ay, az = axes

    tf = ((d0*ax[0], d1*ay[0], d2*az[0], ox),
          (d0*ax[1], d1*ay[1], d2*az[1], oy),
          (d0*ax[2], d1*ay[2], d2*az[2], oz))

    tf_inv = invert_matrix(tf)

    # Replace array by tuples
    tf_inv = tuple(map(tuple, tf_inv))

    return tf, tf_inv


# -----------------------------------------------------------------------------
# Apply scaling and skewing transformations.
#
def scale_and_skew(ijk, step, cell_angles):
    xa, ya, za = skew_axes(cell_angles)

    i, j, k = ijk
    d0, d1, d2 = step
    x, y, z = i*d0, j*d1, k*d2

    xyz = tuple(x*xa[a] + y*ya[a] + z*za[a] for a in (0, 1, 2))
    return xyz


# -----------------------------------------------------------------------------
#
def apply_rotation(r, v):
    rv = [r[a][0]*v[0] + r[a][1]*v[1] + r[a][2]*v[2] for a in (0, 1, 2)]
    return tuple(rv)


# -----------------------------------------------------------------------------
#
def map_point(p, tf):
    tfp = [0, 0, 0]
    for r in range(3):
        tfr = tf[r]
        tfp[r] = tfr[0]*p[0] + tfr[1]*p[1] + tfr[2]*p[2] + tfr[3]
    tfp = tuple(tfp)
    return tfp


# -----------------------------------------------------------------------------
#
class Grid_Subregion(Grid_Data):
    def __init__(self, grid_data, ijk_min, ijk_max, ijk_step=(1, 1, 1)):
        self.full_data = grid_data

        ijk_min = map(lambda a, s: ((a + s - 1) / s) * s, ijk_min, ijk_step)
        self.ijk_offset = ijk_min
        self.ijk_step = ijk_step

        size = map(lambda a, b, s: max(0, (b - a + s) / s),
                   ijk_min, ijk_max, ijk_step)
        origin = grid_data.ijk_to_xyz(ijk_min)
        step = [ijk_step[a]*grid_data.step[a] for a in range(3)]

        Grid_Data.__init__(self, size, grid_data.value_type,
                           origin, step, grid_data.cell_angles,
                           grid_data.rotation, grid_data.symmetries,
                           name=grid_data.name + ' subregion')
        self.rgba = grid_data.rgba
        self.data_cache = None      # Caching done by underlying grid.

    # ---------------------------------------------------------------------------
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        origin, step, size = self.full_region(ijk_origin, ijk_size, ijk_step)
        m = self.full_data.matrix(origin, size, step, progress)
        return m

    # ---------------------------------------------------------------------------
    #
    def cached_data(self, ijk_origin, ijk_size, ijk_step):
        origin, step, size = self.full_region(ijk_origin, ijk_size, ijk_step)
        m = self.full_data.cached_data(origin, size, step)
        return m

    # ---------------------------------------------------------------------------
    #
    def full_region(self, ijk_origin, ijk_size, ijk_step):
        origin = map(lambda i, s, o: (i * s + o),
                     ijk_origin, self.ijk_step, self.ijk_offset)
        size = map(lambda a, b: a * b, ijk_size, self.ijk_step)
        step = map(lambda a, b: a * b, ijk_step, self.ijk_step)
        return origin, step, size

    # ---------------------------------------------------------------------------
    #
    def clear_cache(self):
        self.full_data.clear_cache()


# -----------------------------------------------------------------------------
#
def norm(v):
    from math import sqrt
    d = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return d


# -------------------------------------------- #
#    Array_Grid_Data: wrapper of numpy.array   #
# -------------------------------------------- #
class Array_Grid_Data(Grid_Data):
    def __init__(self, array, origin=(0, 0, 0), step=(1, 1, 1),
                 cell_angles=(90, 90, 90),
                 rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                 symmetries=(),
                 name=''):

        self.array = array

        path = ''
        file_type = ''

        grid_size = list(array.shape)
        grid_size.reverse()

        value_type = array.dtype

        Grid_Data.__init__(self, grid_size, value_type,
                           origin, step, cell_angles=cell_angles,
                           rotation=rotation, symmetries=symmetries,
                           name=name, path=path, file_type=file_type)

        self.writable = True

    # ---------------------------------------------------------------------------
    #
    def read_matrix(self, ijk_origin, ijk_size, ijk_step, progress):
        return self.cached_data(ijk_origin, ijk_size, ijk_step)

    # ---------------------------------------------------------------------------
    #
    def cached_data(self, ijk_origin, ijk_size, ijk_step):
        m = self.matrix_slice(self.array, ijk_origin, ijk_size, ijk_step)
        return m


# -----------------------------------------------------------------------------
# Write an MRC 2000 format file.
#
# Header contains four byte integer or float values:
#
# 1	NX	number of columns (fastest changing in map)
# 2	NY	number of rows
# 3	NZ	number of sections (slowest changing in map)
# 4	MODE	data type :
# 		0	image : signed 8-bit bytes range -128 to 127
# 		1	image : 16-bit halfwords
# 		2	image : 32-bit reals
# 		3	transform : complex 16-bit integers
# 		4	transform : complex 32-bit reals
# 5	NXSTART	number of first column in map
# 6	NYSTART	number of first row in map
# 7	NZSTART	number of first section in map
# 8	MX	number of intervals along X
# 9	MY	number of intervals along Y
# 10	MZ	number of intervals along Z
# 11-13	CELLA	cell dimensions in angstroms
# 14-16	CELLB	cell angles in degrees
# 17	MAP# axis corresp to cols (1,2,3 for X,Y,Z)
# 18	MAPR	axis corresp to rows (1,2,3 for X,Y,Z)
# 19	MAPS	axis corresp to sections (1,2,3 for X,Y,Z)
# 20	DMIN	minimum density value
# 21	DMAX	maximum density value
# 22	DMEAN	mean density value
# 23	ISPG	space group number 0 or 1 (default=0)
# 24	NSYMBT	number of bytes used for symmetry data (0 or 80)
# 25-49   EXTRA	extra space used for anything
# 50-52	ORIGIN  origin in X,Y,Z used for transforms
# 53	MAP	character string 'MAP ' to identify file type
# 54	MACHST	machine stamp
# 55	RMS	rms deviation of map from mean density
# 56	NLABL	number of labels being used
# 57-256 LABEL(20,10) 10 80-character text labels
#
# -----------------------------------------------------------------------------
#
def write_mrc2000_grid_data(grid_data, path, options={}, progress=None):
    mtype = grid_data.value_type.type
    type = closest_mrc2000_type(mtype)

    f = open(path, 'wb')
    if progress:
        progress.close_on_cancel(f)

    header = mrc2000_header(grid_data, type)
    f.write(header)

    stats = Matrix_Statistics()
    isz, jsz, ksz = grid_data.size
    for k in range(ksz):
        matrix = grid_data.matrix((0, 0, k), (isz, jsz, 1))
        if type != mtype:
            matrix = matrix.astype(type)
        f.write(matrix.tostring())
        stats.plane(matrix)
        if progress:
            progress.plane(k)

    # Put matrix statistics in header
    header = mrc2000_header(grid_data, type, stats)
    f.seek(0)
    f.write(header)

    f.close()


# -----------------------------------------------------------------------------
#
def mrc2000_header(grid_data, value_type, stats=None):
    size = grid_data.size

    from numpy import float32, int16, int8, int32
    if value_type == float32:
        mode = 2
    elif value_type == int16:
        mode = 1
    elif value_type == int8:
        mode = 0

    cell_size = map(lambda a, b: a * b, grid_data.step, size)

    if stats:
        dmin, dmax = stats.min, stats.max
        dmean, rms = stats.mean_and_rms(size)
    else:
        dmin = dmax = dmean = rms = 0

    from numpy import little_endian
    if little_endian:
        machst = 0x00004144
    else:
        machst = 0x11110000

    # from chimera.version import release
    ver_stamp = 'Gijs'  # % (release, asctime())
    labels = [ver_stamp[:80]]

    if grid_data.rotation != ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        axis, angle = rotation_axis_angle(grid_data.rotation)
        r = 'Chimera rotation: %12.8f %12.8f %12.8f %12.8f' % (axis + (angle,))
        labels.append(r)

    nlabl = len(labels)
    # Make ten 80 character labels.
    labels.extend(['']*(10-len(labels)))
    labels = [this_label + (80-len(this_label))*'\0' for this_label in labels]
    labelstr = ''.join(labels)

    strings = [
        binary_string(size, int32),  # nx, ny, nz
        binary_string(mode, int32),  # mode
        binary_string((0, 0, 0), int32),  # nxstart, nystart, nzstart
        binary_string(size, int32),  # mx, my, mz
        binary_string(cell_size, float32),  # cella
        binary_string(grid_data.cell_angles, float32),  # cellb
        binary_string((1, 2, 3), int32),  # mapc, mapr, maps
        binary_string((dmin, dmax, dmean), float32),  # dmin, dmax, dmean
        binary_string(0, int32),  # ispg
        binary_string(0, int32),  # nsymbt
        binary_string([0]*25, int32),  # extra
        binary_string(grid_data.origin, float32),  # origin
        'MAP ',  # map
        binary_string(machst, int32),  # machst
        binary_string(rms, float32),  # rms
        binary_string(nlabl, int32),  # nlabl
        labelstr,
        ]

    header = ''.join(strings)
    return header


# -----------------------------------------------------------------------------
#
class Matrix_Statistics:

    def __init__(self):

        self.min = None
        self.max = None
        self.sum = 0.0
        self.sum2 = 0.0

    def plane(self, matrix):

        from numpy import minimum, maximum, add, multiply, array, float32
        matrix_1d = matrix.ravel()
        dmin = minimum.reduce(matrix_1d)
        if self.min is None or dmin < self.min:
            self.min = dmin
        dmax = maximum.reduce(matrix_1d)
        if self.max is None or dmax > self.max:
            self.max = dmax
        self.sum += add.reduce(matrix_1d)
        # TODO: Don't copy array to get standard deviation.
        # Avoid overflow when squaring integral types
        m2 = array(matrix_1d, float32)
        multiply(m2, m2, m2)
        self.sum2 += add.reduce(m2)

    def mean_and_rms(self, size):

        vol = float(size[0])*float(size[1])*float(size[2])
        mean = self.sum / vol
        sdiff = self.sum2 - self.sum*self.sum
        if sdiff > 0:
            from math import sqrt
            rms = sqrt(sdiff) / vol
        else:
            rms = 0
        return mean, rms


# -----------------------------------------------------------------------------
#
def binary_string(values, type):
    from numpy import array
    return array(values, type).tostring()


# -----------------------------------------------------------------------------
#
def closest_mrc2000_type(type):
    from numpy import float, float32, float64
    from numpy import int, int8, int16, int32, int64, character
    from numpy import uint, uint8, uint16, uint32, uint64
    if type in (float32, float64, float, int32, int64, int, uint, uint16,
                uint32, uint64):
        ctype = float32
    elif type in (int16, uint8):
        ctype = int16
    elif type in (int8, character):
        ctype = int8
    else:
        raise TypeError('Volume data has unrecognized type %s' % type)

    return ctype


def convert_numpy_array3d_mrc(data, writename):
    '''USAGE convert_spimage3d_mrc( data_in_numpy_array_float64,
    [outputname].mrc ) This function converts a numpy array (float64)
    to mrc. Writen to convert spimage objects to MRC
    '''
    data = data.real

    if not writename[-4:] == '.mrc':
        writename += '.mrc'
    write_mrc2000_grid_data(Array_Grid_Data(data), writename)


if __name__ == "__main__":
    import sys
    import eke.sphelper
    import numpy

    filename = sys.argv[1]
    writename = sys.argv[2]

    original_image, original_mask = eke.sphelper.import_spimage(
        filename, ["image", "mask"])

    convert_numpy_array3d_mrc(numpy.fft.fftshift(original_image), writename)
