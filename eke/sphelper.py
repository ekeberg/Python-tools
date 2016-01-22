"""Functions to interact with spimage h5py files without loading the spimage package."""
import h5py as _h5py
import numpy as _numpy

def import_spimage(filename, fields=('image',)):
    """Import image part of an spimage file"""

    def _read_image(name):
        """Read an image from a file_handle"""
        if name == "image":
            if file_handle["phased"][0]:
                image = _numpy.squeeze(file_handle['real'][...] + 1.j*file_handle['imag'][...])
            else:
                image = _numpy.real(_numpy.squeeze(file_handle['real'][...]))
        elif name == "mask":
            image = _numpy.bool8(_numpy.squeeze(file_handle["mask"][...]))
        else:
            raise ValueError("Can not load {0}.".format(name))
        return image

    def _read_array(name):
        """Read an array from a file_handle"""
        array = _numpy.squeeze(file_handle[name][...])
        return array

    def _read_single(name):
        """Read an singel value from a file_handle"""
        single = file_handle[name][...][0]
        return single

    allowed_fields = {'image' : _read_image, 'mask' : _read_image, 'detector_distance' : _read_single,
                      'image_center' :_read_array, 'lambda' : _read_single,
                      'num_dimensions' : _read_single, 'phased' : _read_single,
                      'pixel_size' : _read_single, 'scaled' : _read_single,
                      'shifted' : _read_single, 'version' : _read_single}
    field_values = []
    with _h5py.File(filename, 'r') as file_handle:
        for this_field in fields:
            try:
                field_values.append(allowed_fields[this_field](this_field))
            except KeyError:
                raise KeyError("No field named %s" % this_field)
    if len(field_values) == 1:
        return field_values[0]
    else:
        return field_values

def save_spimage(image, filename, mask=None):
    """Create an spimage hdf5 file from an image and optionally a mask."""
    with _h5py.File(filename, 'w') as file_handle:

        file_handle['real'] = _numpy.real(image)
        if abs(_numpy.imag(image)).sum() > 0.:
            file_handle['imag'] = _numpy.imag(image)
            file_handle['phased'] = [1]
        else:
            file_handle['phased'] = [0]

        if mask != None:
            if _numpy.shape(mask) != _numpy.shape(mask):
                raise ValueError("Mask and image have to be the same size")
            file_handle['mask'] = mask
        else:
            file_handle['mask'] = _numpy.ones(_numpy.shape(image), dtype='int32')
        file_handle['detector_distance'] = [0.]
        file_handle['image_center'] = _numpy.array(_numpy.shape(image))/2.-0.5
        file_handle['lambda'] = [0.]
        file_handle['num_dimensions'] = [len(_numpy.shape(image))]
        file_handle['pixel_size'] = [0.]
        file_handle['scaled'] = [0]
        file_handle['shifted'] = [0]
        file_handle['version'] = [2]

        #file_handle.close()

def read_tiff(filename):
    """Read a tiff image."""
    import gdal
    # with gdal.Open(filename) as image_handle:
    #     image = image_handle.ReadAsArray()
    # return image
    image_handle = gdal.Open(filename)
    image = image_handle.ReadAsArray()
    return image
