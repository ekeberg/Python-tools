"""Warning! This is not a full libconfig parser. It can only read very
basic configuration files"""
import re

class Parser(object):
    def __init__(self, filename):
        file_handle = open(filename, 'r')
        self._lines = file_handle.readlines()
        file_handle.close()

    def get_option(self, key):
        for l in self._lines:
            m = re.search("^%s(?:| )=(?:| )(\S.*);" % key, l)
            if m:
                value_string = m.groups()[0]
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
