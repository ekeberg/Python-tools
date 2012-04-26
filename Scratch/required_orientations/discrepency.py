from pylab import *

resolution = arange(2,100) #must be integer
model_side = resolution*2
image_side = resolution*2

Ks = 4.*pi*(resolution)**2/2.
ks = 2.*pi*(resolution)/2.

K = 4./3.*pi*(resolution**3)


model_x = arange(-model_side[-1]/2+0.5, model_side[-1]/2+0.5)
model_y = arange(-model_side[-1]/2+0.5, model_side[-1]/2+0.5)
model_z = arange(-model_side[-1]/2+0.5, model_side[-1]/2+0.5)

mask_sum = zeros(len(resolution))
for i, r in enumerate(resolution):
    mask_sum[i] = sum(model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 < (r)**2)

mask_shell_sum = zeros(len(resolution))
for i, r in enumerate(resolution):
    inner = model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 > (r-1.)**2
    outer = model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 < (r)**2
    
    mask_shell_sum[i] = sum(inner*outer) # * is and
