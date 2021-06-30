import math
import numpy as np
import random
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pandas as pd

"""
JADE算法的仿真测试
首先实现20个测试函数
之后是带存档和不带存档的JADE算法实现函数
"""


# [-100, 100] * 30
def fun_1(x):
    return math.fsum(x ** 2)
    pass


# [-10, 10] * 30
def fun_2(x):
    return math.fsum(np.fabs(x)) + math.prod(np.fabs(x))


# [(-100,100)] * 30    4min13s
def fun_3(x):
    p = 0
    for i in range(len(x)):
        p1 = 0
        for j in range(i + 1):
            p1 += x[j]
            pass
        p += (p1 ** 2)
        pass
    return p
    pass


# [-100, 100] * 30   1min29s   10^-7
def fun_4(x):
    return max(np.fabs(x))
    pass


# [-30, 30] * 30     1min36s 0.66
def fun_5(x):
    p1 = 0
    for i in range(len(x) - 1):
        p1 += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2)
        pass
    return p1
    pass


# [-100, 100] * 30   16s   0.0 -30+
def fun_6(x):
    return np.sum((x + 0.5) ** 2)
    pass


# [-1.28, 1.28] * 30
def fun_7(x):
    p1 = np.arange(len(x)) + 1.0
    return sum(p1 * x ** 4) + np.random.rand()
    pass


# [-500, 500] * 30    18s  10^-3
def fun_8(x):
    p1 = sum(x * np.sin(np.sqrt(np.fabs(x))))
    return 418.98288727243369 * len(x) - p1


# [-5.12, 5.12] * 30   18s 10^-6
def fun_9(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10)
    pass


# [-32, 32] * 30      21s 10^-15
def fun_10(x):
    p1 = -0.2 * np.sqrt(np.sum(x ** 2) / len(x))
    p2 = np.sum(np.cos(2 * np.pi * x)) / len(x)
    return 20 - 20 * math.exp(p1) + math.exp(1) - math.exp(p2)
    pass


# [-600, 600] * 30   19s  0.0
def fun_11(x):
    p1 = np.sqrt(np.arange(len(x)) + 1.0)
    return np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / p1)) + 1.0
    pass


# [-50, 50] * 30    48s  10^-32
def fun_12(x):
    y = np.ones(len(x)) + (x + 1) / 4
    u_x = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 10:
            u_x[i] = 100 * (x[i] - 10) ** 4
            pass
        elif x[i] < -10:
            u_x[i] = 100 * (-x[i] - 10) ** 4
            pass
        else:
            pass
        pass
    p1 = 10 * (np.sin(np.pi * y[0])) ** 2
    p2 = 0
    for i in range(len(x) - 1):
        p2 += (y[i] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * y[i + 1])) ** 2)
        pass
    p3 = np.pi * (p1 + p2 + (y[len(x) - 1]) ** 2) / len(x)
    return p3 + sum(u_x)
    pass


# [-50, 50] * 30    46s  10^-32
def fun_13(x):
    u_x = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > 5:
            u_x[i] = 100 * (x[i] - 5) ** 4
            pass
        elif x[i] < -5:
            u_x[i] = 100 * (-x[i] - 5) ** 4
            pass
        else:
            pass
        pass
    p1 = (np.sin(3 * np.pi * x[0])) ** 2
    p2 = 0
    for i in range(len(x) - 1):
        p2 += (x[i] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[i + 1])) ** 2)
        pass
    p3 = (x[len(x) - 1] - 1) ** 2 * (1 + (np.sin(np.pi * 2 * x[len(x) - 1])) ** 2)
    return 0.1 * (p1 + p2 + p3) + sum(u_x)
    pass


# [(-5, 10), (0, 15)]
def fun_branin(x):
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
    pass


def fun_goldstein_price(x):
    return (1 + ((x[0] + x[1] + 1) ** 2) * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
            30 + ((2 * x[0] - 3 * x[1]) ** 2) * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))


# [(0, 1)] * 3
def fun_hartman3(x):
    alpha = [1.0, 1.2, 3.0, 3.2]
    a = [
        [3, 10, 30],
        [0.1, 10, 35],
        [3, 10, 30],
        [0.1, 10, 35]
    ]
    p = [
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ]
    outer = 0
    for i in range(4):
        inner = 0
        for j in range(3):
            inner += a[i][j] * (x[j] - 0.0001 * p[i][j]) ** 2
            pass
        outer += alpha[i] * np.exp(-inner)
        pass
    return -outer
    pass


# [(0, 1)] * 6
def fun_hartman6(x):
    alpha = [1.0, 1.2, 3.0, 3.2]
    a = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ]
    p = [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ]
    outer = 0
    for i in range(4):
        inner = 0
        for j in range(6):
            inner += a[i][j] * (x[j] - 0.0001 * p[i][j]) ** 2
            pass
        outer += alpha[i] * np.exp(-inner)
        pass
    return -outer
    pass


# [(0, 10)] * 4
def fun_shekel5(x):
    m = 5
    outer = 0
    alpha = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
    c = [
        [4.0, 1, 8, 6, 3, 2, 5, 8, 6, 7.0],
        [4.0, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        [4.0, 1, 8, 6, 3, 2, 5, 8, 6, 7.0],
        [4.0, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]
    ]
    for i in range(m):
        inner = 0
        for j in range(4):
            inner += (x[j] - c[j][i]) ** 2
            pass
        inner += alpha[i] * 0.1
        outer += 1 / inner
        pass
    return -outer
    pass


def fun_shekel7(x):
    m = 7
    outer = 0
    alpha = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
    c = [
        [4.0, 1, 8, 6, 3, 2, 5, 8, 6, 7.0],
        [4.0, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        [4.0, 1, 8, 6, 3, 2, 5, 8, 6, 7.0],
        [4.0, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]
    ]
    for i in range(m):
        inner = 0
        for j in range(4):
            inner += (x[j] - c[j][i]) ** 2
            pass
        inner += alpha[i] * 0.1
        outer += 1 / inner
        pass
    return -outer
    pass


def fun_shekel10(x):
    m = 10
    outer = 0
    alpha = [1, 2, 2, 4, 4, 6, 3, 7, 5, 5]
    c = [
        [4.0, 1, 8, 6, 3, 2, 5, 8, 6, 7.0],
        [4.0, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
        [4.0, 1, 8, 6, 3, 2, 5, 8, 6, 7.0],
        [4.0, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]
    ]
    for i in range(m):
        inner = 0
        for j in range(4):
            inner += (x[j] - c[j][i]) ** 2
            pass
        inner += alpha[i] * 0.1
        outer += 1 / inner
        pass
    return -outer
    pass


def jade_a(fobj, bounds, popsize=100, its=1000, c=0.1):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    mean_cr = 0.5
    mean_mut = 0.5
    a = []  # 定义一个新种群A初始化为空
    for i in range(its):
        s_mut = []
        s_cr = []  # 存储成功的cr值，每代清空
        population = list(population)
        population.sort(key=fobj)
        population = np.array(population)
        best = population[0]
        fitness_best = fobj(best)
        fitness = np.asarray([fobj(ind) for ind in population])
        for j in range(popsize):
            p = 0.05 * popsize
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            idx_x_r2 = random.randint(0, len(population) + len(a) - 3)
            if idx_x_r2 >= (len(population) - 2):
                x_r2 = a[idx_x_r2 - len(population) + 2]
            mut = cauchy.rvs(loc=mean_mut, scale=0.1)
            while mut < 0 or mut > 1:
                if mut < 0:
                    mut = cauchy.rvs(loc=mean_mut, scale=0.1)
                else:
                    mut = 1
            mutant = population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2)
            for mutant_i in range(len(mutant)):
                if mutant[mutant_i] < min_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + min_b[mutant_i]) / 2
                    pass
                elif mutant[mutant_i] > max_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + max_b[mutant_i]) / 2
                    pass
                pass
            cr = np.clip(random.gauss(mean_cr, 0.1), 0, 1)
            cross_points = np.random.rand(dimensions) < cr
            cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            fit = fobj(trial)
            if fit < fitness[j]:
                population_new[j] = trial
                a.append(population[j])
                s_cr.append(cr)
                s_mut.append(mut)
            else:
                population_new[j] = population[j]
        while len(a) > popsize:
            index1 = random.randint(0, len(a) - 1)
            a.pop(index1)
            pass
        for k in range(len(population_new)):
            population[k] = population_new[k]
            pass
        if s_cr:
            mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
            mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        yield best, fitness_best
        pass
    pass


def jade_without_a(fobj, bounds, popsize=100, its=1000, c=0.1):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    mean_cr = 0.5
    mean_mut = 0.5
    for i in range(its):
        s_mut = []
        s_cr = []  # 存储成功的cr值，每代清空
        population = list(population)
        population.sort(key=fobj)
        population = np.array(population)
        best = population[0]
        fitness_best = fobj(best)
        fitness = np.asarray([fobj(ind) for ind in population])
        for j in range(popsize):
            p = 0.05 * popsize
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            mut = cauchy.rvs(loc=mean_mut, scale=0.1)
            while mut < 0 or mut > 1:
                if mut < 0:
                    mut = cauchy.rvs(loc=mean_mut, scale=0.1)
                else:
                    mut = 1
            mutant = population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2)
            for mutant_i in range(len(mutant)):
                if mutant[mutant_i] < min_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + min_b[mutant_i]) / 2
                    pass
                elif mutant[mutant_i] > max_b[mutant_i]:
                    mutant[mutant_i] = (population[j][mutant_i] + max_b[mutant_i]) / 2
                    pass
                pass
            cr = np.clip(random.gauss(mean_cr, 0.1), 0, 1)
            cross_points = np.random.rand(dimensions) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            fit = fobj(trial)
            if fit < fitness[j]:
                population_new[j] = trial
                s_cr.append(cr)
                s_mut.append(mut)
            else:
                population_new[j] = population[j]
        for k in range(len(population_new)):
            population[k] = population_new[k]
            pass
        if s_cr:
            mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
            mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        yield best, fitness_best
        pass
    pass


def jade_test(fun, bounds, its=1000, log=1):
    it = list(jade_a(fun, bounds, popsize=100, its=its))
    print(it[-1])
    x, f = zip(*it)
    plt.plot(f, label='jade')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()
    pass


def jade_a_test_20(fun, bounds, its):
    result = []
    for num in range(50):
        it = list(jade_a(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['JADE', fun.__name__, its, i] for i in result])
    data.to_csv('data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['JADE', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv('data_mean.csv', mode='a', index=False, header=False)
    pass


def jade_without_a_test_20(fun, bounds, its):
    result = []
    for num in range(50):
        it = list(jade_without_a(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['JADE without archive', fun.__name__, its, i] for i in result])
    data.to_csv('data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['JADE without archive', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv('data_mean.csv', mode='a', index=False, header=False)


dic1 = {
    1: {1: fun_1, 2: [(-100, 100)] * 30, 3: 1500},
    2: {1: fun_2, 2: [(-10, 10)] * 30, 3: 2000},
    3: {1: fun_3, 2: [(-100, 100)] * 30, 3: 5000},
    4: {1: fun_4, 2: [(-100, 100)] * 30, 3: 5000},
    5: {1: fun_5, 2: [(-30, 30)] * 30, 3: 3000},
    6: {1: fun_6, 2: [(-100, 100)] * 30, 3: 100},
    7: {1: fun_7, 2: [(-1.28, 1.28)] * 30, 3: 3000},
    8: {1: fun_8, 2: [(-500, 500)] * 30, 3: 1000},
    9: {1: fun_9, 2: [(-5.12, 5.12)] * 30, 3: 1000},
    10: {1: fun_10, 2: [(-32, 32)] * 30, 3: 500},
    11: {1: fun_11, 2: [(-600, 600)] * 30, 3: 500},
    12: {1: fun_12, 2: [(-50, 50)] * 30, 3: 500},
    13: {1: fun_13, 2: [(-50, 50)] * 30, 3: 500},
    14: {1: fun_branin, 2: [(-5, 10), (0, 15)], 3: 200},
    15: {1: fun_goldstein_price, 2: [(-2, 2)] * 2, 3: 200},
    16: {1: fun_hartman3, 2: [(0, 1)] * 3, 3: 200},
    17: {1: fun_hartman6, 2: [(0, 1)] * 6, 3: 200},
    18: {1: fun_shekel5, 2: [(0, 10)] * 4, 3: 200},
    19: {1: fun_shekel7, 2: [(0, 10)] * 4, 3: 200},
    20: {1: fun_shekel10, 2: [(0, 10)] * 4, 3: 200}
}

if __name__ == '__main__':
    for test_index in range(16, 17):
        jade_a_test_20(dic1[test_index][1], dic1[test_index][2], dic1[test_index][3])
        pass
