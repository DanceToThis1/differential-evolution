import matplotlib.pyplot as plt
import datetime
from paper.shade import *


def rastrigin_shade_test_20(fun, bounds):
    result = []
    for num in range(20):
        it = list(shade(fun, bounds, popsize=100, its=3000))
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


def shade_test(fun, bounds, its=3000, goal=0):
    start = datetime.datetime.now()
    it = list(shade(fun, bounds, popsize=100, its=its, goal=goal))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='shade')
    plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with shade')
    plt.show()
    pass
