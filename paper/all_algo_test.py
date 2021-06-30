from functions import *
from paper.de import *
from paper.jde import *
from paper.jade import *
from paper.shade import *
from paper.code import *
import os

path1 = os.path.abspath('.')
path2 = os.path.abspath('..')
"""
字典用来保存待优化函数和对应的限定范围、迭代次数、种群大小、图像是否需要对y轴取对数这些数据。
"""
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

"""
draw_all_algo_performance函数
输入为待优化函数和相应的限定范围
输出为六种算法对该函数的优化结果图像。
"""


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


"""
generate_all_algo_test_data_csv函数
记录每次优化的结果和每50次优化的平均值和标准差，将数据写入csv文件。
"""


def generate_all_algo_test_data_csv():
    for INDEX in range(1, 21):
        jde_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        sade_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        jade_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        shade_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
        code_test_50(dic1[INDEX][1], dic1[INDEX][2], dic1[INDEX][3])
    pass


"""
主函数：对上面两个函数进行测试。
"""
if __name__ == '__main__':
    # generate_all_algo_test_data_csv()
    for index in range(1, 21):
        draw_all_algo_performance(dic1[index][1], dic1[index][2], dic1[index][3], popsize=dic1[index][4], log=dic1[index][5])
