import spimage as _spimage

def image_from_array(image, mask=None):
    if len(image.shape) == 2:
        img = _spimage.sp_image_alloc(image.shape[0], image.shape[1], 1)
    elif len(image.shape) == 3:
        img = _spimage.sp_image_alloc(image.shape[0], image.shape[1], image.shape[2])
    else:
        raise ValueError("Array must be 2 or 3 dimensional")
    img.image[:] = image
    if mask != None:
        img.mask[:] = int32(mask)
    else:
        img.mask[:] = 1
    img.shifted = 0
    img.phased = int(bool(iscomplex(image).sum() > 0))
    return img

def allocate_image(shape):
    if len(shape) == 2:
        img = _spimage.sp_image_alloc(shape[0], shape[1], 1)
    elif len(image.shape) == 3:
        img = _spimage.sp_image_alloc(shape[0], shape[1], shape[2])
    else:
        raise ValueError("Array must be 2 or 3 dimensional")
    return img

def get_basic_phaser(amplitudes, support):
    """This function is not finished"""
    amplitudes_sp = image_from_array(amplitudes)
    beta = spimage.sp_smap_alloc(1)
    spimage.sp_smap_insert(beta, 0, 0.9)
    phase_alg = spimage.sp_phasing_hio_alloc(beta, spimage.SpComplexObject)
    support_sp = image_from_array(support)
    sup_alg = spimage.sp_support_array_init(spimage.sp_support_static_alloc(), 20)

    phaser = spimage.sp_phaser_alloc()
    spimage.sp_phaser_init(phaser, phase_alg, sup_alg, spimage.SpEngineCUDA)
    spimage.sp_phaser_set_amplitudes(phaser, amplitudes_sp)
    spimage.sp_phaser_init_model(phaser, None, spimage.SpModelRandomPhases)
    spimage.sp_phaser_init_support(phaser, support_sp, 0, 0)
    
    return phaser

def smap(input):
    import collections
    if not isinstance(input, collections.Iterable):
        input = [(0, input)]

    smap = _spimage.sp_smap_alloc(len(input))
    for i in input:
        smap.sp_smap_insert(smap, i[0], i[1])

    return smap

def get_constraints(real=False, positive=False):
    if real and positive:
        return _spimage.SpPositiveRealObject
    elif real:
        return _spimage.SpRealObject
    elif positive:
        return _spimage.SpPositiveComplexObject
    else:
        return _spimage.SpNoConstraints
    
def algorithm_hio(beta, real=False, positive=False):
    beta_sp = smap(beta)
    constraints = get_constraints(real, positive)
    algorithm = _spimage.sp_phasing_hio_alloc(beta_sp, constraints)
    return algorithm
    
def algorithm_raar(beta, real=False, positive=False):
    beta_sp = smap(beta)
    constraints = get_constraints(real, positive)
    algorithm = _spimage.sp_phasing_raar_alloc(beta_sp, constraints)
    return algorithm

def algorithm_er(real=False, positive=False):
    constraints = get_constraints(real, positive)
    algorithm = _spimage.sp_phasing_er_alloc(constraints)
    return algorithm

def support_static():
    algorithm = _spimage.sp_support_array_init(_spimage.sp_support_static_alloc(), 20)
    return algorithm

def support_area(area, blur_radius):
    area_sp = smap(area)
    blur_radius_sp = smap(blur_radius)
    algorithm = _spimage.sp_support_array_init(_spimage.sp_support_area_alloc(blur_radius_sp, area_sp), 20)

def support_threshold(threshold, blur_radius):
    threshold_sp = smap(threshold)
    blur_radius_sp = smap(blur_radius)
    algorithm = _spimage.sp_support_array_init(_spimage.sp_support_threshold_alloc(blur_radius_sp, area_sp), 20)

