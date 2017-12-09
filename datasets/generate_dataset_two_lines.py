#/usr/bin/env python3
## -*- coding: utf-8 -*-

a1 = 1
a2 = -0.4

b1 = 3
b2 = 5

data_points = [a1*x+b1 for x in np.arange(-5,5, 0.1)] + [a2*x+b2 for x in
                                                         np.arange(5,15, 0.1)]

data_points = [x+random()*3 for x in data_points]

with open('two_lines.pickle', 'wb') as f:
        pickle.dump(data_points, f)
