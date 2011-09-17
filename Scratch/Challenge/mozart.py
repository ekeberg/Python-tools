from pylab import *

image = imread('mozart.gif')


image_straight_r = image[:,:,0].flatten()
image_straight_g = image[:,:,1].flatten()
image_straight_b = image[:,:,2].flatten()

divided_r = image_straight_r/51
divided_g = image_straight_g/51
divided_b = image_straight_b/51

