

class MyFunction:
    def __init__(self):
        self._counter = 0

    def __call__(self, value):
        self._counter += 1
        return self._counter*value

    def __add__(self, other_function):
        ret = MyFunction()
        ret._counter = self._counter + other_function._counter
        return ret

    def __mul__(self, other_function):
        ret = MyFunction()
        ret._counter = self._counter * other_function._counter
        return ret

    def __neg__(self):
        ret = MyFunction()
        ret._counter = -self._counter
        return ret

    def __sub__(self, other_function):
        neg_other = -other_function
        ret = self+neg_other
        return ret

f1 = MyFunction()
f2 = MyFunction()

print f1(1)
print f1(1)
print f1(1)
print f2(1)
print f2(1)

f3 = f1+f2
print f3(1)
f4 = f1*f2
print f4(1)
f5 = f1-f2
print f5(1)
