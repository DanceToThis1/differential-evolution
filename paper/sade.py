import numpy as np
import pandas as pd
import random
import statistics
import datetime
import matplotlib.pyplot as plt


def sade(fobj, bounds=None, popsize=20, its=1000, goal=0):
    if bounds is None:
        bounds = [(-100, 100)] * 30
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = population[best_idx]
    sp = strategy_probability = [0.25] * 4
    lp = learning_period = 5
    success_memory = np.zeros([lp, 4])
    failure_memory = np.zeros([lp, 4])
    cr_memory = [[], [], [], []]
    cr = 0.5
    for i in range(lp):
        for j in range(popsize):
            popj = population[j]
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]
            strategy_num = -1
            mut = random.gauss(0.5, 0.3)
            cr = random.gauss(0.5, 0.1)
            if (cr < 0) or (cr > 1):
                cr = random.gauss(0.5, 0.1)
            rand_sp = np.random.rand()
            if rand_sp < sp[0]:
                strategy_num = 0
                trial = rand_1_bin(a, b, c, mut, min_b, max_b, popj, dimensions, cr)
            elif rand_sp < sum(sp[:2]):
                strategy_num = 1
                trial = rand_to_best_2_bin(a, b, c, d, mut, min_b, max_b, popj, dimensions, best, cr)
            elif rand_sp < sum(sp[:3]):
                strategy_num = 2
                trial = rand_2_bin(a, b, c, d, e, mut, min_b, max_b, popj, dimensions, cr)
            else:
                strategy_num = 3
                trial = current_to_rand_1(a, b, c, popj, mut, min_b, max_b)
            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                cr_memory[strategy_num].append(cr)
                success_memory[i, strategy_num] += 1
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
            else:
                failure_memory[i, strategy_num] += 1
    for i in range(its):
        success_sum = pd.DataFrame(success_memory).sum(axis=0)
        failure_sum = pd.DataFrame(failure_memory).sum(axis=0)
        skg = success_sum / failure_sum + 0.01
        sp = skg / sum(skg)
        success_memory[(i % lp)] = 0
        failure_memory[(i % lp)] = 0
        for j in range(popsize):
            popj = population[j]
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]
            strategy_num = -1
            rand_sp = np.random.rand()
            mut = random.gauss(0.5, 0.3)
            if rand_sp < sp[0]:
                strategy_num = 0
                cr_median_0 = statistics.median(cr_memory[0])
                cr_0 = random.gauss(cr_median_0, 0.1)
                while (cr_0 < 0) or (cr_0 > 1):
                    cr_0 = random.gauss(cr_median_0, 0.1)
                trial = rand_1_bin(a, b, c, mut, min_b, max_b, popj, dimensions, cr_0)
            elif rand_sp < sum(sp[:2]):
                strategy_num = 1
                cr_median_1 = statistics.median(cr_memory[1])
                cr_1 = random.gauss(cr_median_1, 0.1)
                while (cr_1 < 0) or (cr_1 > 1):
                    cr_1 = random.gauss(cr_median_1, 0.1)
                trial = rand_to_best_2_bin(a, b, c, d, mut, min_b, max_b, popj, dimensions, best, cr_1)
            elif rand_sp < sum(sp[:3]):
                strategy_num = 2
                cr_median_2 = statistics.median(cr_memory[2])
                cr_2 = random.gauss(cr_median_2, 0.1)
                while (cr_2 < 0) or (cr_2 > 1):
                    cr_2 = random.gauss(cr_median_2, 0.1)
                trial = rand_2_bin(a, b, c, d, e, mut, min_b, max_b, popj, dimensions, cr_2)
            else:
                strategy_num = 3
                trial = current_to_rand_1(a, b, c, popj, mut, min_b, max_b)
            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                cr_memory[strategy_num].append(cr)
                success_memory[i % lp, strategy_num] += 1
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
            else:
                failure_memory[i % lp, strategy_num] += 1
        if np.fabs(min(fitness) - goal) < 1e-8:
            print(i)
            break
        yield best, fitness[best_idx]


def rand_1_bin(a, b, c, mut, min_b, max_b, popj, dimensions, cr):
    mutant = np.clip(a + mut * (b - c), min_b, max_b)
    cross_points = np.random.rand(dimensions) < cr
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True
    trial = np.where(cross_points, mutant, popj)
    return trial


def rand_to_best_2_bin(a, b, c, d, mut, min_b, max_b, popj, dimensions, best, cr):
    mutant = np.clip(popj + mut * (best - popj) + mut * (a - b) + mut * (c - d), min_b, max_b)
    cross_points = np.random.rand(dimensions) < cr
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True
    trial = np.where(cross_points, mutant, popj)
    return trial


def rand_2_bin(a, b, c, d, e, mut, min_b, max_b, popj, dimensions, cr):
    mutant = np.clip(a + mut * (b - c) + mut * (d - e), min_b, max_b)
    cross_points = np.random.rand(dimensions) < cr
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True
    trial = np.where(cross_points, mutant, popj)
    return trial


def current_to_rand_1(a, b, c, popj, mut, min_b, max_b):
    k = np.random.rand()
    trial = np.clip(popj + k * (a - popj) + mut * (b - c), min_b, max_b)
    return trial



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


def sade_test(fun, bounds, its=3000, goal=0, log=0):
    start = datetime.datetime.now()
    it = list(sade(fun, bounds, popsize=100, its=its, goal=goal))
    print(it[-1])
    end = datetime.datetime.now()
    print(end - start)
    x, f = zip(*it)
    plt.plot(f, label='sade')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    # plt.savefig('rastrigin with sade')
    plt.show()
    pass