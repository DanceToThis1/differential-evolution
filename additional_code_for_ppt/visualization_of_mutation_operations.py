# -*- coding: utf-8 -*-
"""
变异操作可视化
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
x, y = np.meshgrid(x, y)
z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

fig = plt.figure()
ax = fig.add_subplot()

ax.set_title('mutation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis([-5, 5, -5, 5])
ax.text(-1.3, -2.5, r'$x_{r1}(-1,-2)$', fontsize=14)
ax.text(-5, -2.5, r'$x_{r2}(-3,-2)$', fontsize=14)
ax.text(-1.8, 0.8, r'$x_{r3}(-2,1)$', fontsize=14)
ax.text(-4, -0.3, 'Difference vector')
ax.text(-1, -3, 'Base vector')
ax.text(-0.3, -0.3, r'$v_{i}(-0.5,-0.5)$', fontsize=14)
ax.text(-4, 3, r'$v_i=x_{r1}+F\cdot (x_{r2}-x_{r3})$', fontsize=16)
plt.plot(np.arange(-3, -1, 0.1), [-2] * len(np.arange(-3, -1, 0.1)), '--b')
plt.plot(np.arange(-2.5, -0.5, 0.1), [-0.5] * len(np.arange(-2.5, -0.5, 0.1)), '--b')

ax.arrow(-3, -2, 1, 3, width=0.05)
ax.arrow(-1, -2, 0.5, 1.5, width=0.05)
ax.contour(x, y, z, levels=10, alpha=0.3)
# plt.savefig('contour line test')
plt.show()
