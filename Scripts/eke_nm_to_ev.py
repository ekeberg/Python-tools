#!/bin/env python
import sys
import os
from optparse import OptionParser

sys.path.append('%s/Python/Modules' % os.path.expanduser('~'))

from eke import conversions

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <wavelength in nm>")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print "Must provide an argument"
        exit(1)

    try:
        argument = float(args[0])
    except ValueError:
        print "Invalid argument, can't convert to float"
        exit(1)

    print str(conversions.nm_to_ev(argument))+" eV"