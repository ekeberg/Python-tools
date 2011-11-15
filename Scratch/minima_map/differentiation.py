from pylab import *

side = 10
i = 3

p = random(side)
u = random(side*2)
u.dtype='complex128'

ft_u = fft(u)
abs_ft_u = abs(ft_u)

for i in range(side):
    u_i = zeros(side)
    u_i[i] = 1.
    ft_u_i = fft(u_i)

    abs_operator = conj(ft_u/abs_ft_u)
    print sum((abs_ft_u - p) * real(abs_operator*ft_u_i)) / sum(p**2)


