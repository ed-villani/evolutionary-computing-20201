from random import randint, random, sample, uniform

import numpy as np

MAX_CAPACITY = 35
objs = np.array([
    [10, 5],
    [18, 8],
    [12, 7],
    [14, 6],
    [13, 9],
    [11, 5],
    [8, 4],
    [6, 3]]
)


class BagProblem:
    @staticmethod
    def init_items():
        return np.array([randint(0, 1) for _ in objs])

    @staticmethod
    def current_cap(obj, items_in_bag):
        return items_in_bag @ obj.T[0]

    @staticmethod
    def value_in_bag(obj, items_in_bag):
        return items_in_bag @ obj.T[1]

    @staticmethod
    def penalty_function(obj, items_in_bag):
        def penalty_proportion(obj):
            return (obj.T[1] / obj.T[0])[np.argmax(obj.T[1] / obj.T[0])]

        cap = BagProblem.current_cap(obj, items_in_bag)
        if cap > MAX_CAPACITY:
            return penalty_proportion(obj) * (cap - MAX_CAPACITY)
        else:
            return 0

    @staticmethod
    def fitness_function(obj, items_in_bag):
        return BagProblem.value_in_bag(obj, items_in_bag) - BagProblem.penalty_function(obj, items_in_bag)

    @staticmethod
    def cross(population):
        cand = sample(range(len(objs)), 5)
        cand.sort()

        pais_cross = [[0] * len(objs), [0] * len(objs)]
        for i in range(0, 2):
            pais_cross[i] = population[cand[i]]
        return np.array(pais_cross)

    @staticmethod
    def sort_population_by_fit(population, fitenss_per_been):
        k = np.vstack((population.T, fitenss_per_been)).T
        k = k[np.argsort(-k[:, -1])]

        fitenss_per_been = k[:, -1]
        population = k[:, :-1]
        return fitenss_per_been, population


def main():
    num_population = 6
    max_iters = len(objs) * num_population

    population = np.array([BagProblem.init_items() for _ in range(num_population)])
    fitenss_per_been = np.array([BagProblem.fitness_function(objs, p) for p in population])

    fitenss_per_been, population, = BagProblem.sort_population_by_fit(population, fitenss_per_been)


    total_fitness = sum(fitenss_per_been)
    proportion_fit = fitenss_per_been / total_fitness
    print(proportion_fit)
    current_member = 0
    selected = []
    while current_member < 2:
        random_number = uniform(0, 1)
        current_min = 0
        for i in range(len(proportion_fit)):
            if False:
                current_max = 1
            else:
                current_max = proportion_fit[i] + current_min
            if current_min < random_number < current_max:
                selected.append(population[i])
                print(f"Current Member: {current_member}")
                print(f"Random: {random_number}")
                print(f"Min: {current_min}")
                print(f"Max: {current_max}")
                print(i)
            current_min = proportion_fit[i]
        current_member = current_member + 1
    print(selected)
    # Add fit info por pop


if __name__ == '__main__':
    main()
