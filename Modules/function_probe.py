"""
To probe a functions variables, create a container function and pass it as a variable
to the function. Also add the following line at the point you which to probe.
"exec container._get_code("container")"
"""


class container(object):
    def __init__(self):
        code = """
        l = locals().copy()
        for k in l:"""
    def _get_code(self, name):
        return """
if %s:
    l = locals().copy()
    for k in l:
        if k != "%s":
            exec '%s.%s
        """ % (name, name, name, "%s = %s' % (k, k)")
    def _reset(self):
        variables = self.__dict__.copy()
        for variable in variables:
            exec "del self.%s" % variable

