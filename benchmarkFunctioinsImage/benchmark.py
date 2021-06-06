import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_pic_3D(x, y, z, title, z_min, z_max, offset):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # rstride代表row行步长  cstride代表colum列步长  camp 渐变颜色
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), color='orangered')
    ax.contour(x, y, z, offset=offset, colors='green')
    ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    plt.show()


def get_xy(x_min, x_max, y_min, y_max):
    x = np.arange(x_min, x_max, 0.1)
    y = np.arange(y_min, y_max, 0.1)
    x, y = np.meshgrid(x, y)  # 生成网格点坐标矩阵
    return x, y


def Himmelblau(z_min=0, z_max=800, offset=-2000):
    x, y = get_xy(-5, 5, -5, 5)
    z = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return x, y, z, 'Himmelblau', z_min, z_max, offset


def Six_hump_camel_back(z_min=-25, z_max=5, offset=-25):
    x, y = get_xy(-1.9, 1.9, -1.1, 1.1)
    z = -4 * ((4 - 2.1 * (x ** 2) + ((x ** 4) / 3)) * (x ** 2) + x * y + (4 * (y ** 2) - 4) * (y ** 2))
    return x, y, z, 'Six_HUmp Camel Back', z_min, z_max, offset


def Shubert(z_min=-300, z_max=200, offset=-300):
    x, y = get_xy(-10, 10, -10, 10)
    z1 = (1 * np.cos((1 + 1) * x + 1)) + (2 * np.cos((2 + 1) * x + 2)) + (3 * np.cos((3 + 1) * x + 3)) + (
            4 * np.cos((4 + 1) * x + 4)) + (5 * np.cos((5 + 1) * x + 5))
    z2 = (1 * np.cos((1 + 1) * y + 1)) + (2 * np.cos((2 + 1) * y + 2)) + (3 * np.cos((3 + 1) * y + 3)) + (
            4 * np.cos((4 + 1) * y + 4)) + (5 * np.cos((5 + 1) * y + 5))
    z = -(z1 * z2)
    return x, y, z, 'Shubert', z_min, z_max, offset


def Vincent(z_min=-1, z_max=1, offset=-1):
    x, y = get_xy(0.25, 10, 0.25, 10)
    z1 = np.sin(10 * np.log(x))
    z2 = np.sin(10 * np.log(y))
    z = (z1 + z2) / 2
    return x, y, z, 'Vincent', z_min, z_max, offset


def Sphere(z_min=0, z_max=30, offset=0):
    x, y = get_xy(-3, 3, -3, 3)
    z = x ** 2 + y ** 2
    return x, y, z, "Sphere function", z_min, z_max, offset


def Grienwank(z_min=0, z_max=3, offset=0):
    x, y = get_xy(-10, 10, -10, 10)
    z = (x ** 2) / 4000 + (y ** 2) / 4000 - np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2)) + 1
    return x, y, z, "Grienwank's function", z_min, z_max, offset


def Rastrigin(z_min=0, z_max=100, offset=0):
    x, y = get_xy(-5.52, 5.12, -5.12, 5.12)
    z = 2 * 10 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y)
    return x, y, z, "Rastrigin function", z_min, z_max, offset


def Weierstrass(z_min=-5, z_max=5, offset=-5):
    x, y = get_xy(-2, 2, -2, 2)
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
    x, y = get_xy(-2, 2, -2, 2)
    f2 = 100 * ((x ** 2) - y) ** 2 + (x - 1) ** 2
    z = 1 + (f2 ** 2) / 4000 - np.cos(f2)
    return x, y, z, "F8F2", z_min, z_max, offset


if __name__ == '__main__':
    dic_fun = {
        1: Himmelblau,
        2: Six_hump_camel_back,
        3: Shubert,
        4: Vincent,
        5: Grienwank,
        6: Rastrigin,
        7: Weierstrass,
        8: EF8F2
    }
    x1, y1, z_0, title1, z_min1, z_max1, offset1 = dic_fun[4]()
    draw_pic_3D(x1, y1, z_0, title1, z_min1, z_max1, offset1)
