"""Warning! This is not a full libconfig parser. It can only read very
basic configuration files"""
import re

class Parser(object):
    """Read libconfig file and provide an interface to read simple options."""
    def __init__(self, filename):
        file_handle = open(filename, 'r')
        self._lines = file_handle.readlines()
        file_handle.close()

    def get_option(self, key):
        """Return the value of the option."""
        for line in self._lines:
            match_object = re.search("^%s(?:| )=(?:| )(\S.*);" % key, line)
            if match_object:
                value_string = match_object.groups()[0]
                value_m = re.search('"(.*)"', value_string)
                if value_m:
                    # we have a string
                    value = value_m.groups()[0]
                    return value
                value_m = re.search("^([0-9]+)$", value_string)
                if value_m:
                    #we have an int
                    value = int(value_m.groups()[0])
                    return value
                value_m = re.search("^(false|true)$", value_string)
                if value_m:
                    # we have a boolean
                    if value_m.groups()[0] is "true":
                        return True
                    else:
                        return False
                # we probably have a float then
                return float(value_string)
        raise IOError("No entry matching key: %s" % key)
