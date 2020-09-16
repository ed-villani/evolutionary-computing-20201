from random import randint, sample, uniform

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
    [6, 3]
]
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
            return 2*max((obj.T[1] / obj.T[0]))

        cap = BagProblem.current_cap(obj, items_in_bag)
        if cap > MAX_CAPACITY:
            return penalty_proportion(obj) * (cap - MAX_CAPACITY)
        else:
            return 0

    @staticmethod
    def fitness_function(obj, items_in_bag):
        fit = BagProblem.value_in_bag(obj, items_in_bag) - BagProblem.penalty_function(obj, items_in_bag)
        return fit if fit > 0 else 0

    @staticmethod
    def cross(population):
        cand = sample(range(len(objs)), 5)
        cand.sort()

        pais_cross = [[0] * len(objs), [0] * len(objs)]
        for i in range(0, 2):
            pais_cross[i] = population[cand[i]]
        return np.array(pais_cross)

    @staticmethod
    def sort_population_by_fit(population, fitness_per_being):
        k = np.vstack((population.T, fitness_per_being)).T
        k = k[np.argsort(-k[:, -1])]

        fitness_per_being = k[:, -1]
        population = k[:, :-1]
        return fitness_per_being, population

    @staticmethod
    def sort_population_by_capacity(population, fitness_per_being):
        k = np.vstack((population.T, fitness_per_being, BagProblem.current_cap(objs, population))).T
        k = k[np.argsort(k[:, -1])]
        fitness_per_being = k[:, -2]
        population = k[:, :-2]
        return fitness_per_being, population

    @staticmethod
    def fitness_roulette_selector(n_been, population, fitness_per_being):
        def roulette_selector_per_been(population, fitness_per_being):
            total_fitness = sum(fitness_per_being)
            proportion_fit = fitness_per_being / total_fitness
            picker = uniform(0, 1)
            current = 0
            for been, fit in zip(population, proportion_fit):
                current += fit
                if current > picker:
                    return been

        selected_parents = np.array(
            [roulette_selector_per_been(population, fitness_per_being) for _ in range(int(n_been))])
        return np.array_split(np.array(selected_parents), len(selected_parents) / 2)

    @staticmethod
    def crossover(parents):
        def action(parent_1, parent_2):
            prob_crossover = uniform(0, 1)
            if prob_crossover > 0.6:
                return [parent_1, parent_2]
            cut_point = randint(0, len(parent_1))
            children_1 = np.concatenate((parent_1[: cut_point], parent_2[cut_point:]))
            children_2 = np.concatenate((parent_2[: cut_point], parent_1[cut_point:]))
            return [children_1, children_2]

        return np.array([action(*parent) for parent in parents]).reshape(len(parents) * 2, len(parents[0][0]))

    @staticmethod
    def bit_flip(childrens):
        def action(children):
            for index, bit in enumerate(children):
                if uniform(0, 1) < 0.02:
                    if bit == 1:
                        children[index] = 0
                    elif bit == 0:
                        children[index] = 1
            return children

        return np.array([action(children) for children in childrens])


def main():
    num_population = 1000
    max_iters = 10000
    population = np.array([BagProblem.init_items() for _ in range(num_population)])
    max_gen_to_converge = 10
    last_max_fit = 0
    hit = 0

    for i in range(max_iters):
        fitness_per_being = np.array([BagProblem.fitness_function(objs, p) for p in population])
        index = np.argmax(fitness_per_being)
        print(f"Generation {i}")
        print(f"Better Fitness {fitness_per_being[index]}")
        print(f"Better Being: {population[index]}")
        print(f"Current Capacity: {BagProblem.current_cap(objs, population[index])}")
        print(f"Current Value: {BagProblem.value_in_bag(objs, population[index])}\n")

        if hit >= max_gen_to_converge:
            break
        if (last_max_fit == fitness_per_being[index] or last_max_fit < fitness_per_being[index]) and BagProblem.current_cap(objs, population[index]) <= MAX_CAPACITY:
            hit += 1
            print(f'Hit: {hit}')
        else:
            hit = 0
        last_max_fit = fitness_per_being[index]
        proportion = 0.1
        # Drop worst values
        fitness_per_being, population = BagProblem.sort_population_by_fit(population, fitness_per_being)
        population = population[:int(len(population) * proportion)]
        fitness_per_being = fitness_per_being[:int(len(fitness_per_being) * proportion)]
        # Drops the heavies
        # fitness_per_being, population = BagProblem.sort_population_by_capacity(population, fitness_per_being)
        # population = population[:int(len(population) * proportion)]
        # fitness_per_being = fitness_per_being[:int(len(fitness_per_being) * proportion)]

        selected_parents = BagProblem.fitness_roulette_selector(num_population, population, fitness_per_being)

        childrens = BagProblem.crossover(selected_parents)
        childrens = BagProblem.bit_flip(childrens)
        population = childrens


if __name__ == '__main__':
    main()
