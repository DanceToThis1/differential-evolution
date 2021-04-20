from paper.jde import *
import matplotlib.pyplot as plt
import datetime


def jde_test(fun, bounds, mut=0.9, cr=0.1, its=3000, goal=0, log=0):
    start = datetime.datetime.now()
    it = list(jde(fun, bounds, mut=mut, cr=cr, popsize=100, its=its, goal=goal))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='jde')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with jde')
    plt.show()
    pass
