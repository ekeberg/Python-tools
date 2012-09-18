
import sys
import os
from optparse import OptionParser

sys.path.append('%s/Python/Modules' % os.path.expanduser('~'))

import conversions

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <energy in eV>")
    (options, args) = parser.parse_args()

    if len(args) < 1:
        print "Must provide an argument"
        exit(1)

    try:
        argument = float(args[0])
    except ValueError:
        print "Invalid argument, can't convert to float"
        exit(1)

    print str(conversions.ev_to_nm(argument))+" nm"

