import matplotlib.pyplot as plt
import datetime
from paper.code import *


def code_test(fun, bounds, mut=0.9, cr=0.1, its=3000, goal=0, log=0):
    start = datetime.datetime.now()
    it = list(code(fun, bounds, popsize=100, its=its, goal=goal))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='code')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()
    pass

