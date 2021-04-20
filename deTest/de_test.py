import matplotlib.pyplot as plt
import datetime
from paper.de import *


def de_rand_1_test(fun, bounds, mut=0.9, cr=0.1, its=3000, goal=0):
    start = datetime.datetime.now()
    it = list(de(fun, bounds, mut, cr, popsize=100, its=its, goal=goal))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='de rand 1 bin')
    # plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with de_rand_1')
    plt.show()
    pass
