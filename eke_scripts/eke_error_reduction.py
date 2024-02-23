from __future__ import print_function
import numpy as _numpy
import spimage as _spimage
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="Diffraction pattern")
    parser.add_argument("real_space", help="Starting real-space model")
    parser.add_argument("support", help="Support")
    parser.add_argument("-n", "--number_of_iterations", type=int, default=100,
                        help="Number of iterations of ER")
    parser.add_argument("-o", "--outdir", default=".",
                        help="Output directory")
    parser.add_argument("-a", "--affix", default="refined",
                        help="This name will be added to all output files.")
    args = parser.parse_args()

    _numpy.random.seed()
    _spimage.sp_srand(_numpy.random.randint(1e6))
    intensities = _spimage.sp_image_read(args.pattern, 0)
    if intensities.shifted == 0:
        amplitudes = _spimage.sp_image_shift(intensities)
    else:
        amplitudes = _spimage.sp_image_duplicate(intensities,
                                                 _spimage.SP_COPY_ALL)
    _spimage.sp_image_dephase(amplitudes)
    _spimage.sp_image_to_amplitudes(amplitudes)

    real_space = _spimage.sp_image_read(args.real_space, 0)
    support = _spimage.sp_image_read(args.support, 0)

    phase_alg = _spimage.sp_phasing_er_alloc(_spimage.SpNoConstraints)

    sup_alg = _spimage.sp_support_array_init(
        _spimage.sp_support_static_alloc(),
        20)

    # create phaser
    phaser = _spimage.sp_phaser_alloc()
    _spimage.sp_phaser_init(phaser, phase_alg, sup_alg, _spimage.SpEngineCUDA)
    _spimage.sp_phaser_set_amplitudes(phaser, amplitudes)
    _spimage.sp_phaser_init_model(phaser, real_space, 0)
    _spimage.sp_phaser_init_support(phaser, support, 0, 0)

    fourier_space = _spimage.sp_image_ifftw3(real_space)

    support = _numpy.bool8(support.image)
    ereal_start = _numpy.sqrt((abs(real_space.image[~support])**2).sum() /
                              (abs(real_space.image)**2).sum())

    mask = _numpy.bool8(amplitudes.mask)
    efourier_start = _numpy.sqrt(
        ((abs(fourier_space.image[mask])
          - abs(amplitudes.image[mask]))**2).sum()
        / ((abs(amplitudes.image[mask])**2).sum()
           + (abs(fourier_space.image[~mask])**2).sum()))

    _spimage.sp_phaser_iterate(phaser, args.number_of_iterations)

    model_out = _spimage.sp_phaser_model(phaser)
    support_out = _spimage.sp_phaser_support(phaser)
    fmodel_out = _spimage.sp_phaser_fmodel(phaser)
    real_space_end = _spimage.sp_phaser_model_before_projection(phaser)
    fourier_space_end = _spimage.sp_phaser_fmodel(phaser)

    ereal_end = _numpy.sqrt((abs(real_space_end.image[~support])**2).sum()
                            / (abs(real_space_end.image)**2).sum())
    efourier_end = _numpy.sqrt(
        ((abs(fourier_space_end.image[mask])
          - abs(amplitudes.image[mask]))**2).sum()
        / ((abs(amplitudes.image[mask])**2).sum()
           + (abs(fourier_space_end.image[~mask])**2).sum()))

    _spimage.sp_image_write(
        model_out, f"{args.outdir}/real_space-{args.affix}.h5", 0)
    _spimage.sp_image_write(
        support_out, f"{args.outdir}/support-{args.affix}.h5", 0)
    _spimage.sp_image_write(
        fmodel_out, f"{args.outdir}/fourier_space-{args.affix}.h5", 0)
    print("Ereal:    %g -> %g" % (ereal_start, ereal_end))
    print("Efourier: %g -> %g" % (efourier_start, efourier_end))


if __name__ == "__main__":
    main()
