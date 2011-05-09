
import sys
import os

sys.path.append('%s/Python/Modules' % os.path.expanduser('~'))

import conversions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Must provide an argument"
        exit(1)

    try:
        argument = float(sys.argv[1])
    except ValueError:
        print "Invalid argument, can't convert to float"
        exit(1)

    print conversions.ev_to_nm(argument)
