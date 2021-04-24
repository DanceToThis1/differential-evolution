from paper.sade import *


def code(fobj, bounds, popsize=20, its=2000, goal=0):
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
        if np.fabs(min(fitness) - goal) < 1e-8:
            print(i)
            break
        yield best, fitness[best_idx]
    pass