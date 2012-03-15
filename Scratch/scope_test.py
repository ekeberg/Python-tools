
class container(object):
    def __init__(self):
        code = """
        l = locals().copy()
        for k in l:"""
    def _get_code(self):
        return """
if variable_container:
    l = locals().copy()
    for k in l:
        if k != "var_cont":
            exec "var_cont.%s = %s" % (k,k)
        """

def store_locals(container):
    if container:
        l = super.locals().copy()
        for k in l:
            if l[k] != container:
                exec "container.%s = %s" % (k,k)

def function(var_cont=None):
    x = 3
    y = 5
    exec variable_container.get_code()
    return x*y

c = container()
