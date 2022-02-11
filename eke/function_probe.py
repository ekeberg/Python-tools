"""To probe a functions variables, create a container object and pass
it as a variable to the function. Also add the following line at the
point you which to probe.  "exec <Container>._get_code("<Container>")"
"""


class Container(object):
    """Stores function variables to make them accessable outside the
    function"""
    def __init__(self):
        pass

    @classmethod
    def _get_code(cls, name):
        """Copies all local variables to the class"""
        return """
if %s:
    l = locals().copy()
    for k in l:
        if k != "%s":
            exec '%s.%s
        """ % (name, name, name, "%s = %s' % (k, k)")

    def _reset(self):
        """Delete stored variables"""
        variables = self.__dict__.copy()
        for variable in variables:
            exec("del self.%s" % variable)
