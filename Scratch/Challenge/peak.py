
import pickle
#import pylab
import numpy
import matplotlib

f = open('peak.p','rb')
p = pickle.load(f)
f.close()

for p0 in p:
    for p1 in p0:
        print p1[1], " ",
    print ""

sums = [sum([p1[1] for p1 in p0]) for p0 in p]
values = [[p1[1] for p1 in p0] for p0 in p]

cumulative_values = [[sum(v[:i+1])for i in range(len(v))] for v in values]


screen = numpy.zeros((95,23))
for i,p0 in enumerate(p):
    for j,p1 in enumerate(p0):
        if p1[0] == ' ':
            screen[cumulative_values[i][j]-1,i] = 1.0
        else:
            screen[cumulative_values[i][j]-1,i] = 2.0

#matplotlib.imshow(screen)
#show()
for j in range(23):
    for i in range(95):
        print int(screen[i,j]),
    print ''

for i in range(23):
    l = []
    for j in screen[:,i]:
        if j == 0.0:
            l.append(' ')
        elif j == 1.0:
            l.append(' ')
        else:
            l.append('#')
    print ''.join(l)
    #print ''.join([str(int(j))for j in screen[:,i]])
