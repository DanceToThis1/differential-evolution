from functions import *
from paper.de import *
from paper.jde import *
from paper.jade import *
from paper.shade import *
from paper.code import *
import os

path1 = os.path.abspath('.')
path2 = os.path.abspath('..')
dic1 = {
    1: {1: fun_rastrigin, 2: [(-5.12, 5.12)] * 30, 3: 1000, 4: 100, 5: 1},
    2: {1: fun_ackley, 2: [(-5, 5)] * 2, 3: 200, 4: 20, 5: 1},
    3: {1: fun_sphere, 2: [(-100, 100)] * 30, 3: 1000, 4: 100, 5: 1},
    4: {1: fun_rosenbrock, 2: [(-100, 100)] * 30, 3: 2000, 4: 100, 5: 1},
    5: {1: fun_beale, 2: [(-4.5, 4.5)] * 2, 3: 100, 4: 20, 5: 1},
    6: {1: fun_goldstein_price, 2: [(-2, 2)] * 2, 3: 100, 4: 20, 5: 0},
    7: {1: fun_booth, 2: [(-10, 10)] * 2, 3: 100, 4: 20, 5: 1},
    8: {1: fun_bukin_n6, 2: [(-15, -5), (-3, 3)], 3: 200, 4: 20, 5: 1},
    9: {1: fun_matyas, 2: [(-10, 10)] * 2, 3: 100, 4: 20, 5: 1},
    10: {1: fun_levi_n13, 2: [(-10, 10)] * 2, 3: 100, 4: 20, 5: 1},
    11: {1: fun_himmelblau, 2: [(-5, 5)] * 2, 3: 100, 4: 20, 5: 1},
    12: {1: fun_three_hump_camel, 2: [(-5, 5)] * 2, 3: 100, 4: 20, 5: 1},
    13: {1: fun_easom, 2: [(-100, 100)] * 2, 3: 100, 4: 20, 5: 0},
    14: {1: fun_cross_in_tray, 2: [(-10, 10)] * 2, 3: 100, 4: 20, 5: 0},
    15: {1: fun_eggholder, 2: [(-512, 512)] * 2, 3: 100, 4: 20, 5: 0},
    16: {1: fun_holder_table, 2: [(-10, 10)] * 2, 3: 100, 4: 20, 5: 0},
    17: {1: fun_mccormick, 2: [(-1.5, 4), (-3, 4)], 3: 100, 4: 20, 5: 0},
    18: {1: fun_schaffrer_n2, 2: [(-100, 100)] * 2, 3: 100, 4: 20, 5: 1},
    19: {1: fun_schaffrer_n4, 2: [(-100, 100)] * 2, 3: 100, 4: 20, 5: 0},
    20: {1: fun_styblinski_tang, 2: [(-5, 5)] * 30, 3: 1000, 4: 100, 5: 0}
}


def draw_all_algo_performance(fun, bounds, its=1000, popsize=20, log=1):
    it_de = list(de(fun, bounds, popsize=popsize, its=its))
    it_jde = list(jde(fun, bounds, popsize=popsize, its=its))
    it_sade = list(sade(fun, bounds, popsize=popsize, its=its))
    it_jade = list(jade(fun, bounds, popsize=popsize, its=its))
    it_shade = list(shade(fun, bounds, popsize=popsize, its=its))
    it_code = list(code(fun, bounds, popsize=popsize, its=its))
    x, f1 = zip(*it_de)
    x, f2 = zip(*it_jde)
    x, f3 = zip(*it_sade)
    x, f4 = zip(*it_jade)
    x, f5 = zip(*it_shade)
    x, f6 = zip(*it_code)
    plt.xlabel("iterations")
    plt.ylabel("best_fitness_value")
    plt.title(fun.__name__)
    plt.plot(f1, '-', label='de')
    plt.plot(f2, '--', label='jde')
    plt.plot(f3, ':', label='sade')
    plt.plot(f4, dashes=[12, 12], label='jade')
    plt.plot(f5, dashes=[6, 12], label='shade')
    plt.plot(f6, dashes=[3, 6], label='code')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    # plt.savefig(path2 + '/image/all_algo_performance_update/' + str(fun.__name__))
    plt.show()
    pass


def generate_all_algo_test_data_csv():
    for INDEX in range(1, 21):
        jde_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        sade_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        jade_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        shade_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        code_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
    pass


if __name__ == '__main__':
    """生成csv文件，记录每次优化的结果和每50次优化的平均值和标准差"""
    # generate_all_algo_test_data_csv()
    """画出6个算法在测试函数上优化的表现"""
    for index in range(1, 21):
        draw_all_algo_performance(dic1[index][1], dic1[index][2], dic1[index][3], popsize=dic1[index][4], log=dic1[index][5])

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