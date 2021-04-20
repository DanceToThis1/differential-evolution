from benchmarkFunctions.Functions import *
from deTest.sade_test import *
from deTest.de_test import *
from deTest.jde_test import *
from deTest.jade_test import *
from deTest.shade_test import *

if __name__ == '__main__':
    de_rand_1_test(himmelblau, [(-5, 5)] * 2, mut=0.9, cr=0.1, its=100)
    # jde_test(fun_rastrigin, [(-5.12, 5.12)] * 30, mut=0.9, cr=0.1)
    # sade_test(fun_rastrigin, [(-5.12, 5.12)] * 10)
    # jade_test(fun_rastrigin, [(-5.12, 5.12)] * 30)
    # shade_test(fun_rastrigin, [(-5.12, 5.12)] * 30)
    pass
"""
函数名：             全局最小值个数   全局最小值   维度限制    建议x取值范围
fun_rastrigin       1              0          无         [-5.12,5.12]
fun_sphere          1              0          无         [-100,100]
fun_grienwank       1              0          无         [-600,600]


"""