from pylab import *

fig = figure(1)
fig.clear()
ax_real = fig.add_subplot(211)
ax_fourier = fig.add_subplot(212)

x = arange(-10., 10, 0.01)
y = abs(mod(x, 2.) - 1.)

#ax_real.plot(x, y)

y_ft = fftshift(fft(y))
x_ft = x/20.*len(x)/max(x)

ax_fourier.plot(x_ft, abs(y_ft))


#frequencies = len(x)/2 + array([0, -10, 10, -30, 30, -50, 50, -70, 70, -90, 90])
frequencies = len(x)/2 + arange(-200,200,10)

total_comp = zeros(shape(x))
for i in frequencies:
    freq_comp = 1.0/float(len(x))*y_ft[i]*exp(1.j*pi*x_ft[i]*x)
    total_comp += freq_comp
    ax_real.plot(real(freq_comp))

ax_real.plot(real(total_comp))

ax_fourier.set_xlim([-10., 10.])

show()
