from paper.sade import *
import matplotlib.pyplot as plt
import os

path1 = os.path.abspath('.')
path2 = os.path.abspath('..')

"""
变异向量越界有新的处理方式还没加
不确定迭代次数的设计是否有问题。
"""


def code(fobj, bounds, popsize=20, its=2000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population = min_b + pop * diff
    population_new = np.random.rand(popsize, dimensions)
    for i in range(len(population_new)):
        population_new[i] = population[i]
        pass
    fitness = np.asarray([fobj(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best = population[best_idx]
    param_dic = {
        1: [1.0, 0.1],
        2: [1.0, 0.9],
        3: [0.8, 0.2]
    }
    # its = FES / 3
    for i in range(popsize, its * 3, 3):
        for j in range(popsize):
            popj = population[j]
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c, d, e = population[np.random.choice(idxs, 5, replace=False)]
            rand_1, rand_2, current_1 = np.random.randint(1, 4, 3)
            trial_rand1 = rand_1_bin(a, b, c, param_dic[rand_1][0], min_b, max_b, popj, dimensions,
                                     param_dic[rand_1][1])
            trial_rand2 = rand_2_bin(a, b, c, d, e, param_dic[rand_2][0], min_b, max_b, popj, dimensions,
                                     param_dic[rand_2][1])
            trial_current = current_to_rand_1(a, b, c, popj, param_dic[current_1][0], min_b, max_b)
            fit = [fobj(trial_rand1), fobj(trial_rand2), fobj(trial_current)]
            b_index = np.argmin(fit)
            best_trial = [trial_rand1, trial_rand2, trial_current][b_index]
            f = fit[b_index]
            if f < fitness[j]:
                fitness[j] = f
                population_new[j] = best_trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = best_trial
                    pass
                pass
            else:
                population_new[j] = population[j]
                pass
            pass
        for k in range(len(population_new)):
            population[k] = population_new[k]
        yield best, fitness[best_idx]
    pass


def code_test(fun, bounds, its=3000, log=0):
    it = list(code(fun, bounds, popsize=100, its=its))
    print(it[-1])
    x, f = zip(*it)
    plt.plot(f, label='code')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()
    pass


def code_test_50(fun, bounds, its):
    result = []
    for num in range(50):
        it = list(code(fun, bounds, popsize=100, its=its))
        result.append(it[-1][-1])
        print(num, result[-1])
        pass
    data = pd.DataFrame([['CODE', fun.__name__, its, i] for i in result])
    data.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    data_mean = pd.DataFrame([['CODE', fun.__name__, its, mean_result, std_result]])
    data_mean.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', index=False, header=False)
    pass
