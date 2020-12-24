import numpy as np


l = 8
n = 30
m = 3

w = np.round(np.random.rand(30) * 64.0, 6)
s = np.random.randint(1, 9, (30, 3))
d = np.random.randint(1, 4, (30, 3))

with open('ada-final-public/11.in', 'w') as f:
    f.write('{}\n'.format(l))
    f.write('{}\n'.format(n))
    for i in range(n):
        f.write('{}\n'.format(m))
        f.write('{}\n'.format(w[i]))
        f.write('{} {} {}\n'.format(s[i][0], d[i][0], 0))
        f.write('{} {} {} {}\n'.format(s[i][1], d[i][1], 1, 1))
        f.write('{} {} {} {} {}\n'.format(s[i][2], d[i][2], 2, 1, 2))
        


