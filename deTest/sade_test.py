from paper.sade import *
import datetime
import matplotlib.pyplot as plt


def rastrigin_sade_test_20(fun, bounds):
    result = []
    for num in range(20):
        it = list(sade(fun, bounds, popsize=100, its=3000))
        result.append(it[-1][-1])
        pass
    mean_result = np.mean(result)
    std_result = np.std(result)
    success_num = 0
    for i in result:
        if np.fabs(i - 0) < 1e-5:
            success_num += 1
            pass
        pass
    return mean_result, std_result, success_num


def sade_test(fun, bounds):
    start = datetime.datetime.now()
    it = list(sade(fun, bounds, popsize=100, its=1000))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='sade')
    # plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with sade')
    plt.show()
    # fun_weierstrass
    # FEF8F2 和上一个函数跑起来停不下来。
    ##############################################################
    # mean_result, std_result, success_num = rastrigin_sade_test()
    # print(mean_result, std_result, success_num)
    pass

