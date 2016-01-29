import numpy as _numpy
import spimage as _spimage
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser(usage="%name -n NUMBER_OF_ITERATIONS -f DIFFRACTION_PATTERN -r REAL_SPACE_MODEL -s SUPPORT [-o OUTPUT_DIR -a AFFIX]")
    parser.add_option("-n", action="store", type="int", default=100, dest="number_of_iterations", help="Number of iterations of ER")
    parser.add_option("-f", action="store", type="string", default=None, dest="pattern", help="Diffraction pattern")
    parser.add_option("-r", action="store", type="string", default=None, dest="real_space", help="Starting real-space model")
    parser.add_option("-s", action="store", type="string", default=None, dest="support", help="Support")
    parser.add_option("-o", action="store", type="string", default=".", dest="output_dir", help="Output directory")
    parser.add_option("-a", action="store", type="string", default="refined", dest="output_affix", help="This name will be added to all output files.")
    options, args = parser.parse_args()

    _numpy.random.seed()
    _spimage.sp_srand(_numpy.random.randint(1e6))
    intensities = _spimage.sp_image_read(options.pattern, 0)
    if intensities.shifted == 0:
        amplitudes = _spimage.sp_image_shift(intensities)
    else:
        amplitudes = _spimage.sp_image_duplicate(intensities, _spimage.SP_COPY_ALL)
    _spimage.sp_image_dephase(amplitudes)
    _spimage.sp_image_to_amplitudes(amplitudes)

    real_space = _spimage.sp_image_read(options.real_space, 0)
    support = _spimage.sp_image_read(options.support, 0)

    phase_alg = _spimage.sp_phasing_er_alloc(_spimage.SpNoConstraints)

    sup_alg = _spimage.sp_support_array_init(_spimage.sp_support_static_alloc(), 20)

    # create phaser
    phaser = _spimage.sp_phaser_alloc()
    _spimage.sp_phaser_init(phaser, phase_alg, sup_alg, _spimage.SpEngineCUDA)
    _spimage.sp_phaser_set_amplitudes(phaser, amplitudes)
    _spimage.sp_phaser_init_model(phaser, real_space, 0)
    _spimage.sp_phaser_init_support(phaser, support, 0, 0)

    #real_space_s = _spimage.sp_image_shift(real_space)
    fourier_space = _spimage.sp_image_ifftw3(real_space)

    ereal_start = _numpy.sqrt((abs(real_space.image[~_numpy.bool8(support.image)])**2).sum() / (abs(real_space.image)**2).sum())
    efourier_start = _numpy.sqrt(((abs(fourier_space.image[_numpy.bool8(amplitudes.mask)]) - abs(amplitudes.image[_numpy.bool8(amplitudes.mask)]))**2).sum() / ((abs(amplitudes.image[_numpy.bool8(amplitudes.mask)])**2).sum() + (abs(fourier_space.image[~_numpy.bool8(amplitudes.mask)])**2).sum()))

    _spimage.sp_phaser_iterate(phaser, options.number_of_iterations)

    model_out = _spimage.sp_phaser_model(phaser)
    support_out = _spimage.sp_phaser_support(phaser)
    fmodel_out = _spimage.sp_phaser_fmodel(phaser)
    real_space_end = _spimage.sp_phaser_model_before_projection(phaser)
    fourier_space_end = _spimage.sp_phaser_fmodel(phaser)

    ereal_end = _numpy.sqrt((abs(real_space_end.image[~_numpy.bool8(support.image)])**2).sum() / (abs(real_space_end.image)**2).sum())
    efourier_end = _numpy.sqrt(((abs(fourier_space_end.image[_numpy.bool8(amplitudes.mask)]) - abs(amplitudes.image[_numpy.bool8(amplitudes.mask)]))**2).sum() / ((abs(amplitudes.image[_numpy.bool8(amplitudes.mask)])**2).sum() + (abs(fourier_space_end.image[~_numpy.bool8(amplitudes.mask)])**2).sum()))

    _spimage.sp_image_write(model_out, "%s/real_space-%s.h5" % (options.output_dir, options.output_affix), 0)
    _spimage.sp_image_write(support_out, "%s/support-%s.h5" % (options.output_dir, options.output_affix), 0)
    _spimage.sp_image_write(fmodel_out, "%s/fourier_space-%s.h5" % (options.output_dir, options.output_affix), 0)
    print "Ereal:    %g -> %g" % (ereal_start, ereal_end)
    print "Efourier: %g -> %g" % (efourier_start, efourier_end)

