from pylab import *
import re

image = imread('mozart.gif')


image_straight_r = image[:,:,0].flatten()
image_straight_g = image[:,:,1].flatten()
image_straight_b = image[:,:,2].flatten()

divided_r = image_straight_r/51
divided_g = image_straight_g/51
divided_b = image_straight_b/51

r_char = ''.join([chr(i) for i in image_straight_r])
g_char = ''.join([chr(i) for i in image_straight_g])
b_char = ''.join([chr(i) for i in image_straight_b])
expr1 = re.compile('\xff{5}(.)')
expr2 = re.compile('\x00{5}(.)')
r_m = expr1.finditer(r_char)
g_m = expr2.finditer(g_char)
b_m = expr1.finditer(b_char)

r_pos = [m.span()[1] for m in r_m]
g_pos = [m.span()[1] for m in g_m]
b_pos = [m.span()[1] for m in b_m]

r_m = expr1.finditer(r_char)
g_m = expr2.finditer(g_char)
b_m = expr1.finditer(b_char)

r_value = [ord(m.groups()[0]) for m in r_m]
g_value = [ord(m.groups()[0]) for m in g_m]
b_value = [ord(m.groups()[0]) for m in b_m]

f = open('mozart.gif','rb')
binary_image = ''.join(f.readlines())
f.close()
