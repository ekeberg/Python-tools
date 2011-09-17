import pylab
import re

evil_file = open('evil2.gfx')
lines = evil_file.readlines()
evil_file.close()

all = ''.join(lines)
expr = re.compile('([a-zA-Z]{4,100})')

p = expr.findall(all)

image = pylab.imread('evil1.jpg')
image_r = image[:,:,0]
image_g = image[:,:,1]
image_b = image[:,:,2]

fig = pylab.figure(1)
fig.clear()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(image_r,origin='lower')
ax2.imshow(image_g,origin='lower')
ax3.imshow(image_b,origin='lower')
pylab.show()

for i in range(5):
    pile = all[i::5]
    f = open("evil_file_%d.jpg" % i,'wb')
    f.write(pile)
    f.close()
