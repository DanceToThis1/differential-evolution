from paper.jade import *
import matplotlib.pyplot as plt
import datetime


def rastrigin_jade_test_20(fun, bounds):
    result = []
    for num in range(20):
        it = list(jade(fun, bounds, popsize=100, its=3000))
        result.append(it[-1][-1])
        pass
    mean_result = np.mean(result)
    std_result = np.std(result)
    success_num = 0
    for i in result:
        if np.fabs(i - 0) < 1e-8:
            success_num += 1
            pass
        pass
    return mean_result, std_result, success_num


def jade_test(fun, bounds):
    start = datetime.datetime.now()
    it = list(jade(fun, bounds, popsize=100, its=3000))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='jade')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with jade')
    plt.show()
    # fun_weierstrass
    # FEF8F2 和上一个函数跑起来停不下来。
    ##############################################################
    # mean_result, std_result, success_num = rastrigin_jade_test()
    # print(mean_result, std_result, success_num)
    pass

