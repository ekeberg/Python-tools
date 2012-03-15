from pylab import *

side = 100
object_side = 20
intensity = 10.
gaussian_noise = 0.01
threshold = 1.

img = zeros((side, side))
img[side/2-object_side/2:side/2+object_side/2,side/2-object_side/2:side/2+object_side/2] = intensity

pattern = fft2(img)/side + normal(0. , gaussian_noise, (side, side))
phases = angle(pattern)
amplitudes = poisson(abs(pattern))
mask = amplitudes <= threshold
pattern_masked = maximum(amplitudes-threshold, zeros((side, side))) * exp(1.j*phases)
#pattern_masked = pattern*(amplitudes > threshold)
#pattern_masked = amplitudes * exp(1.j*phases)
img_masked = ifft2(pattern_masked)*side

print sum(abs(abs(img_masked)*sum(img)/sum(abs(img_masked)) - img))
