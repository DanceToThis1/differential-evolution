from benchmarkFunctions.Functions import *
from deTest.sade_test import *
from deTest.de_test import *
from deTest.jde_test import *
from deTest.jade_test import *
from deTest.shade_test import *

if __name__ == '__main__':
    fun_index = 18
    algo_index = 1
    print('function index = ' + str(fun_index))
    print('algorithm index = ' + str(algo_index))
    algorithm_dic = {
        1: de_rand_1_test,
        2: jde_test,
        3: sade_test,
        4: jade_test,
        5: shade_test
    }
    functions_dic = {
        1: fun_rastrigin,
        2: fun_ackley,
        3: fun_sphere,
        4: fun_rosenbrock,
        5: fun_beale,
        6: fun_goldstein_price,
        7: fun_booth,
        8: fun_bukin_n6,
        9: fun_matyas,
        10: fun_levi_n13,
        11: fun_himmelblau,
        12: fun_three_hump_camel,
        13: fun_easom,
        14: fun_cross_in_tray,
        15: fun_eggholder,
        16: fun_holder_table,
        17: fun_mccormick,
        18: fun_schaffrer_n2,
        19: fun_schaffrer_n4,
        20: fun_styblinski_tang
    }
    bounds_dic = {
        1: [(-5.12, 5.12)] * 30,
        2: [(-5, 5)] * 2,
        3: [(-100, 100)] * 30,
        4: [(-100, 100)] * 30,
        5: [(-4.5, 4.5)] * 2,
        6: [(-2, 2)] * 2,
        7: [(-10, 10)] * 2,
        8: [(-15, -5), (-3, 3)],
        9: [(-10, 10)] * 2,
        10: [(-10, 10)] * 2,
        11: [(-5, 5)] * 2,
        12: [(-5, 5)] * 2,
        13: [(-100, 100)] * 2,
        14: [(-10, 10)] * 2,
        15: [(-512, 512)] * 2,
        16: [(-10, 10)] * 2,
        17: [(-1.5, 4), (-3, 4)],
        18: [(-100, 100)] * 2,
        19: [(-100, 100)] * 2,
        20: [(-5, 5)] * 30,
    }
    goal_dic = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 3,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: -1,
        14: -2.06261,
        15: -959.6407,
        16: -19.2085,
        17: -1.9133,
        18: 0,
        19: 0.292579,
        20: -39.16617 * 30
    }
    log_dic = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 0,
        7: 1,
        8: 0,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 0,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 1,
        19: 0,
        20: 0
    }
    algorithm_dic[algo_index](functions_dic[fun_index], bounds_dic[fun_index], its=3000, goal=goal_dic[fun_index],
                              log=log_dic[fun_index])
    pass

"""
benchmark functions
  函数名                       最小值点                              x范围                  备注
1 fun_rastrigin           f(0,0,...,0) = 0                     [(-5.12, 5.12)] * n       
2 fun_ackley              f(0,0) = 0                           [(-5, 5)] * 2
3 fun_sphere              f(0,0,...,0) = 0                     [(-100, 100)] * n
4 fun_rosenbrock          f(1,1,...,1) = 0                     [(-100, 100)] * n         n>=2
5 fun_beale               f(3, 0.5) = 0                        [(-4.5, 4.5)] * 2
6 fun_goldstein_price     f(0, -1) = 3                         [(-2, 2)] * 2
7 fun_booth               f(1, 3) = 0                          [(-10, 10)] * 2
8 fun_bukin_n6            f(-10, 1) = 0                        [(-15, -5), (-3, 3)]      
9 fun_matyas              f(0,0,...,0) = 0                     [(-10, 10)] * n           n=2
10 fun_levi_n13           f(1, 1) = 0                          [(-10, 10)] * 2
11 fun_himmelblau         f(3.0,2.0) = 0                       [(-5, 5)] * 2
                          f(-2.805118, 3.131312) = 0
                          f(-3.779310, -3.283186) = 0
                          f(3.5834428, -1.848126) = 0
12 fun_three_hump_camel   f(0, 0) = 0                          [(-5, 5)] * 2
13 fun_easom              f(pi,pi,...pi) = -1                  [(-100,100)] * n          n=2
14 fun_cross_in_tray      f(+-1.34941, +-1.34941) = -2.06261   [(-10, 10)] * n           n=2
15 fun_eggholder          f(512,404.2319) = -959.6407          [(-512,512)] * 2 
16 fun_holder_table       f(+-8.05502, +-9.66459) = -19.2085   [(-10,10)] * 2
17 fun_mccormick          f(-0.54719, 1.54719) = -1.9133       [(-1.5, 4), (-3, 4)]
18 fun_schaffrer_n2       f(0,0) = 0                           [(-100,100)] * 2
19 fun_schaffrer_n4       f(0,+-1.25313) = 0.292579            [(-100,100)] * 2
20 fun_styblinski_tang    -39.16617*n < f(-2.903534,           [(-5,5)] * n
                          ...,-2.903534) < -39.16616*n
                          1174.9851 n = 30
"""
