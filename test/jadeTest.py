import datetime
import numpy as np
import random
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pandas as pd


#                           times
# [-100, 100] * 30          1500*20  1.7646305234857702e-52   7.6918422560958e-52
def fun_1(x):
    return np.sum(x ** 2)
    pass


# [-10, 10] * 30
def fun_2(x):
    return np.sum(np.fabs(x)) + np.prod(np.fabs(x))


# [(-100,100)] * 30
def fun_3(x):
    p = 0
    for i in range(len(x)):
        p1 = 0
        for j in range(i):
            p1 += x[j]
            pass
        p += (p1 ** 2)
        pass
    return p
    pass


# [-100, 100] * 30   16s   10^-7
def fun_4(x):
    return np.max(np.fabs(x))
    pass


# [-30, 30] * 30     30s   10^1
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


# [-1.28, 1.28] * 30  20s  0.8
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
    p1 = -0.2 * np.sqrt(sum(x ** 2) / len(x))
    p2 = np.sum(np.cos(2 * np.pi * x)) / len(x)
    return np.exp(1) + 20 - 20 * np.exp(p1) - np.exp(p2)
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


def jade(fobj, bounds, popsize=20, its=1000, c=0.1):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    mut = 0.5
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
            p = 0.05 * popsize  # p的设置，固定值0.05-0.2 * NP，或者自适应调整。
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
            mutant = np.clip(population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2), min_b, max_b)
            cr = random.gauss(mean_cr, 0.1)
            cross_points = np.random.rand(dimensions) < cr
            if not np.any(cross_points):
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
            index = random.randint(0, len(a) - 1)
            a.pop(index)
            pass
        for k in range(len(population_new)):
            population[k] = population_new[k]
            pass
        if s_cr:
            mean_cr = (1 - c) * mean_cr + c * np.mean(s_cr)
            mean_mut = (1 - c) * mean_mut + c * (sum(ff ** 2 for ff in s_mut) / sum(s_mut))
        yield best, fitness_best


def jade_test(fun, bounds, its=1000, log=1):
    start = datetime.datetime.now()
    it = list(jade(fun, bounds, popsize=100, its=its))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='jade')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()
    pass


def jade_test_20(fun, bounds, its):
    result = []
    for num in range(2):
        it = list(jade(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['jade', fun.__name__, its, i] for i in result])
    data.to_csv('data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['JADE', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv('data_mean.csv', mode='a', header=False)


def shade(fobj, bounds, popsize=20, its=1000, h=100):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    mcr = [0.5] * h
    mf = [0.5] * h
    m = 0
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
        fk = []
        for j in range(popsize):
            r_i = random.randint(0, h - 1)
            cr = random.gauss(mcr[r_i], 0.1)
            mut = cauchy.rvs(loc=mf[r_i], scale=0.1)
            while mut < 0 or mut > 1:
                if mut < 0:
                    mut = cauchy.rvs(loc=mf[r_i], scale=0.1)
                else:
                    mut = 1
            p = random.randint(int(0.05 * popsize), int(0.2 * popsize))  # p的设置，固定值0.05-0.2 * NP，或者随机调整。
            idx_x_best_p = random.randint(0, int(p))
            x_best_p = population[idx_x_best_p]
            idxs = [idx for idx in range(popsize) if idx != j]
            x_r1, x_r2 = population[np.random.choice(idxs, 2, replace=False)]
            idx_x_r2 = random.randint(0, len(population) + len(a) - 3)
            if idx_x_r2 >= (len(population) - 2):
                x_r2 = a[idx_x_r2 - len(population) + 2]
            mutant = np.clip(population[j] + mut * (x_best_p - population[j]) + mut * (x_r1 - x_r2), min_b, max_b)
            cross_points = np.random.rand(dimensions) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            fit = fobj(trial)
            if fit < fitness[j]:
                population_new[j] = trial
                a.append(population[j])
                s_cr.append(cr)
                s_mut.append(mut)
                fk.append(np.abs(fit - fitness[j]))  # 保存差值，差值越高后面占比越高。
                pass
            else:
                population_new[j] = population[j]
                pass
            pass
        while len(a) > popsize:
            index = random.randint(0, len(a) - 1)
            a.pop(index)
            pass
        for k in range(len(population_new)):
            population[k] = population_new[k]
        if s_cr:
            w_k = [fki / sum(fk) for fki in fk]
            mean_wa = sum(w_k[index] * s_cr[index] for index in range(len(s_cr)))
            mean_wl1 = sum(w_k[index] * s_mut[index] ** 2 for index in range(len(s_cr)))
            mean_wl2 = sum(w_k[index] * s_mut[index] for index in range(len(s_cr)))
            mean_wl = mean_wl1 / mean_wl2
            mcr[m] = mean_wa
            mf[m] = mean_wl
            m += 1
            if m >= h:
                m = 1
                pass
            pass
        else:
            m += 1
            if m >= h:
                m = 1
                pass
            pass
        yield best, fitness_best


def shade_test_20(fun, bounds, its):
    result = []
    for num in range(2):
        it = list(shade(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['shade', fun.__name__, its, i] for i in result])
    data.to_csv('data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['SHADE', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv('data_mean.csv', mode='a', header=False)


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
    14: {1: fun_5, 2: [(-30, 30)] * 30, 3: 20000},
    15: {1: fun_6, 2: [(-100, 100)] * 30, 3: 1500},
    16: {1: fun_8, 2: [(-500, 500)] * 30, 3: 9000},
    17: {1: fun_9, 2: [(-5.12, 5.12)] * 30, 3: 5000},
    18: {1: fun_10, 2: [(-32, 32)] * 30, 3: 2000},
    19: {1: fun_11, 2: [(-600, 600)] * 30, 3: 3000},
    20: {1: fun_12, 2: [(-50, 50)] * 30, 3: 1500},
    21: {1: fun_13, 2: [(-50, 50)] * 30, 3: 1500}
}
for test_index in range(13):
    shade_test_20(dic1[test_index + 1][1], dic1[test_index + 1][2], dic1[test_index + 1][3])
    pass
