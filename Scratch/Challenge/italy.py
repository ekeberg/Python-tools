from pylab import *

DIR_X_POS = 0; DIR_Y_POS = 1; DIR_X_NEG = 2; DIR_Y_NEG = 3

image = imread('wire.png')


image_out = zeros((100,100,3))

x_max = 100; y_max = 100
x_min = 0; y_min = 0
x = 0; y = 0
direction = 0

for pixel_value in image.squeeze():
    image_out[x,y,:] = pixel_value
    if direction == DIR_X_POS:
        if x < x_max-1:
            x += 1
        else:
            direction = DIR_Y_POS
            y_min += 1
            y += 1
    elif direction == DIR_Y_POS:
        if y < y_max-1:
            y += 1
        else:
            direction = DIR_X_NEG
            x_max -= 1
            x -= 1
    elif direction == DIR_X_NEG:
        if x > x_min:
            x -= 1
        else:
            direction = DIR_Y_NEG
            y_max -= 1
            y -= 1
    elif direction == DIR_Y_NEG:
        if y > y_min:
            y -= 1
        else:
            direction = DIR_X_POS
            x_min += 1
            x += 1
