from paper.jde import *
import matplotlib.pyplot as plt
import datetime


def jde_test(fun, bounds, mut, cr):
    start = datetime.datetime.now()
    it = list(jde(fun, bounds, mut, cr, popsize=100, its=3000))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='jde')
    # plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with jde')
    plt.show()
    pass

