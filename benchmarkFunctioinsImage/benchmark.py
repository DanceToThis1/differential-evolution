import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_pic(x, y, title, x_min, x_max, y_min, y_max):
    plt.grid(True)  ##增加格点
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.plot(x, y)
    plt.show()


def draw_pic_3D(x, y, z, title, z_min, z_max, offset):
    fig = plt.figure()
    ax = Axes3D(fig)
    # rstride代表row行步长  cstride代表colum列步长  camp 渐变颜色
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), color='orangered')
    # 绘制等高线
    ax.contour(x, y, z, offset=offset, colors='green')
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    plt.show()


def get_x_and_y(x_min, x_max, y_min, y_max):
    x = np.arange(x_min, x_max, 0.1)
    y = np.arange(y_min, y_max, 0.1)
    x, y = np.meshgrid(x, y)  # 生成网格点坐标矩阵
    return x, y


# 1
def Five_uneven_peak_Trap(x_min=0, x_max=30, y_min=0, y_max=200):
    x = np.linspace(0, 30, 1000)

    interval0 = [1 if (i >= 0.0) & (i < 2.5) else 0 for i in x]
    interval1 = [1 if (i >= 2.5) & (i < 5.0) else 0 for i in x]
    interval2 = [1 if (i >= 5.0) & (i < 7.5) else 0 for i in x]
    interval3 = [1 if (i >= 7.5) & (i < 12.5) else 0 for i in x]
    interval4 = [1 if (i >= 12.5) & (i < 17.5) else 0 for i in x]
    interval5 = [1 if (i >= 17.5) & (i < 22.5) else 0 for i in x]
    interval6 = [1 if (i >= 22.5) & (i < 27.5) else 0 for i in x]
    interval7 = [1 if (i >= 27.5) & (i < 30.0) else 0 for i in x]

    y = 80 * (2.5 - x) * interval0 + 64 * (x - 2.5) * interval1 + 64 * (7.5 - x) * interval2 + 28 * (
            x - 7.5) * interval3 + 28 * (17.5 - x) * interval4 + 32 * (x - 17.5) * interval5 + 32 * (
                27.5 - x) * interval6 + 80 * (x - 27.5) * interval7
    return x, y, 'Five-Uneven-Peak Trap', x_min, x_max, y_min, y_max


# 2
def Equal_maxima(x_min=0, x_max=1, y_min=0, y_max=1):
    x = np.linspace(0, 1, 1000)
    y = np.sin(5 * np.pi * x) ** 6
    return x, y, 'Equal Maxima', x_min, x_max, y_min, y_max


# 3
def Uneven_decreasing_maxima(x_min=0, x_max=1, y_min=0, y_max=1):
    x = np.linspace(0, 1, 1000)
    y = np.exp(-2 * np.log(2) * ((x - 0.08) / 0.854) ** 2) * (np.sin(5 * np.pi * (x ** (3 / 4)) - 0.05)) ** 6
    return x, y, 'Uneven Decreasing Maxima', x_min, x_max, y_min, y_max


# 4
def Himmelblau(z_min=0, z_max=800, offset=-2000):
    x, y = get_x_and_y(-5, 5, -5, 5)
    z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return x, y, z, 'Himmelblau', z_min, z_max, offset


# 5
def Six_hump_camel_back(z_min=-25, z_max=5, offset=-25):
    x, y = get_x_and_y(-1.9, 1.9, -1.1, 1.1)
    z = -4 * ((4 - 2.1 * (x ** 2) + ((x ** 4) / 3)) * (x ** 2) + x * y + (4 * (y ** 2) - 4) * (y ** 2))
    return x, y, z, 'Six_HUmp Camel Back', z_min, z_max, offset


# 6   --2D---
def Shubert(z_min=-300, z_max=200, offset=-300):
    x, y = get_x_and_y(-10, 10, -10, 10)
    z1 = (1 * np.cos((1 + 1) * x + 1)) + (2 * np.cos((2 + 1) * x + 2)) + (3 * np.cos((3 + 1) * x + 3)) + (
            4 * np.cos((4 + 1) * x + 4)) + (5 * np.cos((5 + 1) * x + 5))
    z2 = (1 * np.cos((1 + 1) * y + 1)) + (2 * np.cos((2 + 1) * y + 2)) + (3 * np.cos((3 + 1) * y + 3)) + (
            4 * np.cos((4 + 1) * y + 4)) + (5 * np.cos((5 + 1) * y + 5))
    z = -(z1 * z2)
    return x, y, z, 'Shubert', z_min, z_max, offset


# 7 ---2D---
def Vincent(z_min=-1, z_max=1, offset=-1):
    x, y = get_x_and_y(0.25, 10, 0.25, 10)

    z1 = np.sin(10 * np.log(x))
    z2 = np.sin(10 * np.log(y))
    z = (z1 + z2) / 2
    return x, y, z, 'Vincent', z_min, z_max, offset


# 8 ---存在点问题----
def Modified_rastrigin_all_global_optima(z_min=-40, z_max=10, offset=-40):
    x, y = get_x_and_y(0, 1, 0, 1)
    z1 = 10 + 9 * np.cos(2 * np.pi * 3 * x)
    z2 = 10 + 9 * np.cos(2 * np.pi * 4 * y)
    z = -(z1 + z2)
    return x, y, z, 'Modified Rastrigin All Global Optima', z_min, z_max, offset


# Sphere Function
def Sphere(z_min=0, z_max=30, offset=0):
    x, y = get_x_and_y(-3, 3, -3, 3)
    z = x ** 2 + y ** 2
    return x, y, z, "Sphere function", z_min, z_max, offset


# Grienwank's Function
def Grienwank(z_min=0, z_max=3, offset=0):
    x, y = get_x_and_y(-10, 10, -10, 10)
    z = (x ** 2) / 4000 + (y ** 2) / 4000 - np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2)) + 1
    return x, y, z, "Grienwank's function", z_min, z_max, offset


# restrigin function
def Rastrigin(z_min=0, z_max=100, offset=0):
    x, y = get_x_and_y(-5.52, 5.12, -5.12, 5.12)
    z = 2 * 10 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)
    return x, y, z, "Rastrigin function", z_min, z_max, offset


# Weierstrass Function
def Weierstrass(z_min=-5, z_max=5, offset=-5):
    x, y = get_x_and_y(-2, 2, -2, 2)
    a = 0.5
    b = 3
    kmax = 20
    z1, z2, z3 = 0, 0, 0
    for k1 in range(1, kmax + 1):
        z_1 = (a ** k1) * np.cos(2 * np.pi * (b ** k1) * (x + 0.5))
        z_2 = (a ** k1) * np.cos(2 * np.pi * (b ** k1) * (y + 0.5))
        z1 += z_1
        z2 += z_2
    for k2 in range(1, kmax + 1):
        z_3 = (a ** k2) * np.cos(2 * np.pi * (b ** k2) * 0.5)
        z3 += z_3
    z = z1 + z2 - z3
    return x, y, z, "Weierstrass Function", z_min, z_max, offset


# Expanded Griewank's plus Rosenbrock's function (EF8F2)
def EF8F2(z_min=-100, z_max=3000, offset=-100):
    x, y = get_x_and_y(-2, 2, -2, 2)
    f2 = 100 * ((x ** 2) - y) ** 2 + (x - 1) ** 2
    z = 1 + (f2 ** 2) / 4000 - np.cos(f2)
    return x, y, z, "F8F2", z_min, z_max, offset


# Five_uneven_peak_Trap
# x, y, title, x_min, x_max, y_min, y_max = Five_uneven_peak_Trap()

# Equal_maxima
# x, y, title, x_min, x_max, y_min, y_max = Equal_maxima()

# Uneven_decreasing_maxima
# x, y, title, x_min, x_max, y_min, y_max = Uneven_decreasing_maxima()

# draw_pic(x, y, title, x_min, x_max, y_min, y_max)

#################################################################################

# Himmelblau
x, y, z, title, z_min, z_max, offset = Himmelblau()

# Six_hump_camel_back
# x, y, z, title, z_min, z_max, offset = Six_hump_camel_back()

# Shubert
# x, y, z, title, z_min, z_max, offset = Shubert()

# Vincent
# x, y, z, title, z_min, z_max, offset = Vincent()

# Modified_rastrigin_all_global_optima
# x, y, z, title, z_min, z_max, offset = Modified_rastrigin_all_global_optima()

# Grienwank's Function
# x, y, z, title, z_min, z_max, offset = Grienwank()

# Rastrigin Function
# x, y, z, title, z_min, z_max, offset = Rastrigin()

# Weierstrass Function
# x, y, z, title, z_min, z_max, offset = Weierstrass()

# EF8F2
# x, y, z, title, z_min, z_max, offset = EF8F2()

draw_pic_3D(x, y, z, title, z_min, z_max, offset)
