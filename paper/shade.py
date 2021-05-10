"""
SHADE success history based adaptive DE
基于JADE的改进
基于高斯分布和柯西分布随机生成参数，均值由原来记录SCR，SF的均值改为新存储MCR，MF中的随机值。
p 由固定值改为随机值，范围为pmin到0.2，pmin计算方法没看懂暂时用0.05替代
删除参数c，引入参数h，作用与c相似。
"""

import numpy as np
import random
import pandas as pd
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import datetime


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


def shade_test(fun, bounds, its=3000, log=0):
    start = datetime.datetime.now()
    it = list(shade(fun, bounds, popsize=100, its=its))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='shade')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()
    pass


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
