#!/usr/bin/env python
import sys
import pylab
from eke import elements
from eke import conversions
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser(usage="%prog -w WAVELENGTH -e ENERGY -m MATERIAL")
    parser.add_option("-w", action="store", type="float", dest="wavelength",
                      help="Photon wavelength in nm.")
    parser.add_option("-e", action="store", type="float", dest="energy",
                      help="Photon energy in eV (takes presedence over wavelength).")
    parser.add_option("-m", action="store", type="string", dest="material",
                      help="Material. Can be an element or one of water, protein, virus, cell.")
    parser.add_option("-d", action="store", type="float", dest="density",
                      help="If the material is an element, the density has to be provided.")
    (options,argv) = parser.parse_args()
    if not options.material:
        print "Error: A material has to be specified."
        exit(1)
    elif not (options.wavelength or options.energy):
        print "Error: Energy or wavelength has to be specified."
        exit(1)

    if options.energy:
        energy = options.energy
    else:
        energy = conversions.nm_to_ev(options.wavelength)

    if options.material in elements.elements:
        if not options.density:
            print "Error: density has to be provided for this material."
            exit(1)
        kwargs = {options.material : 1}
        attenuation_length = elements.get_attenuation_length(energy,elements.Material(options.density,
                                                                                      **kwargs))
    elif options.material.lower() in elements.materials.keys():
        attenuation_length = elements.get_attenuation_length(energy,elements.materials[options.material.lower()])
    else:
        print "Error: invalid material."
        exit(1)
        
    print "%g um" % (attenuation_length*1e6)

