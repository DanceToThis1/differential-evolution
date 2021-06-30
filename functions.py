import numpy as np

"""
20个测试函数，来自https://en.wikipedia.org/wiki/Test_functions_for_optimization
测试函数要求：实数连续编码
"""


# 1        f(0,0,...,0) = 0   x [-5.12,5.12]*n
def fun_rastrigin(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10)


# 2 ackley  f(0,0,...,0) = 0   x [-5,5]*n
def fun_ackley(x):
    p1 = -0.2 * np.sqrt(sum(x ** 2) / len(x))
    p2 = np.sum(np.cos(2 * np.pi * x)) / len(x)
    return 20 - 20 * np.exp(p1) + np.exp(1) - np.exp(p2)
    pass


# 3 Sphere  f(0,0,...,0) = 0    x [-100,100]*n
def fun_sphere(x):
    return np.sum(x ** 2)


# 4 rosenbrock f(1,1,...,1) = 0  x [-100,100]*n n>=2
def fun_rosenbrock(x):
    p1 = 0
    for i in range(len(x) - 1):
        p1 += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
        pass
    return p1
    pass


# 5 beale f(3,0.5) = 0  x  [-4.5,4.5]*2
def fun_beale(x):
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
            2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    pass


# 6 goldstein price f(0, -1) = 3   x  [-2,2]*2
def fun_goldstein_price(x):
    return (1 + ((x[0] + x[1] + 1) ** 2) * (
            19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
                   30 + ((2 * x[0] - 3 * x[1]) ** 2) * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


# 7 booth f(1,3) = 0  x  [-10,10]*2
def fun_booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


# 8 bukin n.6 f(-10, 1) = 0   x 二维 -15<x1<-5  -3<x2<3
def fun_bukin_n6(x):
    return 100 * np.sqrt(np.fabs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.fabs(x[0] + 10)


# 9 matyas   f(0,0,...,0) = 0  x [-10,10]*n  建议二维
def fun_matyas(x):
    p1 = 0.26 * np.sum(x ** 2)
    p2 = 0.48
    for i in x:
        p2 *= i
        pass
    return p1 - p2


# 10 levi n.13 f(1,1)=0   x [-10,10]*2
def fun_levi_n13(x):
    return (np.sin(3 * np.pi * x[0])) ** 2 + (x[0] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1])) ** 2) + (
            x[1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[1]) ** 2))
    pass


# 11 himmelblau
# Variable ranges: x, y in [-5, 5]
# No. of global peaks: 4
# No. of local peaks:  0
# f(3.0,2.0) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.5834428, -1.848126) = 0
def fun_himmelblau(x=None):
    if x is None:
        return None
    result = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    return result


# 12 three hump camel f(0, 0) = 0    x [-5,5]*2
def fun_three_hump_camel(x):
    return 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2
    pass


# 13   easom     f(pi,pi,...pi) = -1  x [-100,100]*n 建议二维
def fun_easom(x):
    p1 = 1
    p2 = 0
    for i in x:
        p1 *= np.cos(i)
        p2 += (i - np.pi) ** 2
        pass
    return -p1 * np.exp(-p2)
    pass


# 14 cross in tray   f(+-1.34941, +-1.34941) = -2.06261  x [-10, 10]*n 建议二维
def fun_cross_in_tray(x):
    p1 = 1
    for i in x:
        p1 *= np.sin(i)
        pass
    return -0.0001 * (np.fabs(p1 * np.exp(np.fabs(100 - np.sqrt(sum(x ** 2)) / np.pi))) + 1) ** 0.1
    pass


# 15 fun_eggholder   f(512,404.2319) = -959.6407  x [-512,512]*2
def fun_eggholder(x):
    return -(x[1] + 47) * np.sin(np.sqrt(np.fabs(x[0] / 2 + x[1] + 47))) - x[0] * np.sin(
        np.sqrt(np.fabs(x[0] - x[1] - 47)))
    pass


# 16 fun_holder_table f(+-8.05502, +-9.66459) = -19.2085  x [-10,10]*2
def fun_holder_table(x):
    return -(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - np.sqrt(sum(x ** 2)) / np.pi)))
    pass


# 17 fun_mcCormick  f(-0.54719, 1.54719) = -1.9133   -1.5<x1<4  -3<x2<4
def fun_mccormick(x):
    return np.sin(sum(x)) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
    pass


# 18 fun_schaffer_n2   f(0,0) = 0 x[-100,100]*2
def fun_schaffrer_n2(x):
    return 0.5 + ((np.sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5) / ((1 + 0.001 * (sum(x ** 2))) ** 2)
    pass


# 19 fun_schaffer_n4   f(0,+-1.25313) = 0.292579   x [-100,100]*2
def fun_schaffrer_n4(x):
    return 0.5 + ((np.cos(np.sin(np.fabs(x[0] ** 2 - x[1] ** 2)))) ** 2 - 0.5) / ((1 + 0.001 * (sum(x ** 2))) ** 2)
    pass


# 20 fun_styblinski_tang    f(-2.903534,-2.903534,...,-2.903534) = -39.16617 * n 与 -39.16616 * n 之间  x [-5,5]*n
def fun_styblinski_tang(x):
    return sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2
    pass


"""
benchmark functions
  函数名                       最小值点                              x范围                  备注      迭代次数
1 fun_rastrigin           f(0,0,...,0) = 0                     [(-5.12, 5.12)] * n                1000
2 fun_ackley              f(0,0) = 0                           [(-5, 5)] * 2                      200
3 fun_sphere              f(0,0,...,0) = 0                     [(-100, 100)] * n                  1000
4 fun_rosenbrock          f(1,1,...,1) = 0                     [(-100, 100)] * n         n>=2     2000
5 fun_beale               f(3, 0.5) = 0                        [(-4.5, 4.5)] * 2                  100
6 fun_goldstein_price     f(0, -1) = 3                         [(-2, 2)] * 2                      100
7 fun_booth               f(1, 3) = 0                          [(-10, 10)] * 2                    100
8 fun_bukin_n6            f(-10, 1) = 0                        [(-15, -5), (-3, 3)]               200
9 fun_matyas              f(0,0,...,0) = 0                     [(-10, 10)] * n           n=2      100
10 fun_levi_n13           f(1, 1) = 0                          [(-10, 10)] * 2                    100
11 fun_himmelblau         f(3.0,2.0) = 0                       [(-5, 5)] * 2                      100
                          f(-2.805118, 3.131312) = 0
                          f(-3.779310, -3.283186) = 0
                          f(3.5834428, -1.848126) = 0
12 fun_three_hump_camel   f(0, 0) = 0                          [(-5, 5)] * 2                      100
13 fun_easom              f(pi,pi,...pi) = -1                  [(-100,100)] * n          n=2      100
14 fun_cross_in_tray      f(+-1.34941, +-1.34941) = -2.06261   [(-10, 10)] * n           n=2      100
15 fun_eggholder          f(512,404.2319) = -959.6407          [(-512,512)] * 2                   100
16 fun_holder_table       f(+-8.05502, +-9.66459) = -19.2085   [(-10,10)] * 2                     100
17 fun_mccormick          f(-0.54719, 1.54719) = -1.9133       [(-1.5, 4), (-3, 4)]               100
18 fun_schaffrer_n2       f(0,0) = 0                           [(-100,100)] * 2                   100
19 fun_schaffrer_n4       f(0,+-1.25313) = 0.292579            [(-100,100)] * 2                   100
20 fun_styblinski_tang    -39.16617*n < f(-2.903534,           [(-5,5)] * n                       1000
                          ...,-2.903534) < -39.16616*n
                          1174.9851 n = 30
"""
