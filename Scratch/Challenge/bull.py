import pylab

""" what are you looking at?"""
""" len(a[30]) = ?"""

a = [1, 11, 21, 1211, 111221, 'x']


def generate_next(number):
    number_string = str(number)

    new = []
    last = None
    count = 0
    for c in number_string:
        if c != last:
            if count > 0:
                new.append('%d%s' % (count, last))
            
            last = c
            count = 1
        else:
            count += 1
    new.append('%d%s' % (count,last))
    full = ''.join(new)
    return int(full)


a = [1]

for i in range(30):
    a.append(generate_next(a[-1]))
