import warnings
from copy import deepcopy
import numpy as np
from matplotlib import MatplotlibDeprecationWarning
from numpy.random.mtrand import uniform, choice, randint
import matplotlib.pyplot as plt

from tp3.peaks import peaks
from tp3.rastrigin import rastrigin


class DifferentialEvolution:

    def __init__(self, n_pop, axis_range, max_gen=10000, max_gen_to_converge=50, C=None, F=None):
        self._n_pop = n_pop
        self._axis_range = axis_range
        self._max_gen = max_gen
        self._max_gen_to_converge = max_gen_to_converge
        self._c = C
        self._f = F
        self._data_per_gen = None

    def arg_min(self, func, log=False):
        func = func
        n_pop = self._n_pop
        axis_ranges = self._axis_range
        MAX_GEN = self._max_gen
        F = self._f
        C = self._c

        pop = DifferentialEvolution.init_population(axis_ranges, n_pop, 2)
        current_min = np.inf
        max_gen_to_converge = self._max_gen_to_converge
        hit = 0

        data_arg_min = []
        for i in range(MAX_GEN):
            children = DifferentialEvolution.gen_children(pop, C=C, F=F)
            pop = DifferentialEvolution.select_new_generation(pop, children, func)
            fit = DifferentialEvolution.pop_fitness(pop, func)
            if log:
                print(f"Generation {i} - MIN {min(fit)} - MEAN {np.mean(fit)} - MAX {max(fit)}")
            data_arg_min.append([i, min(fit), np.mean(fit), max(fit)])

            if min(fit) < current_min:
                current_min = min(fit)
                hit = 0
            if min(fit) == current_min:
                hit = hit + 1
            if hit > max_gen_to_converge:
                self._data_per_gen = np.array(data_arg_min)
                return np.array(data_arg_min)


    def plot(self):
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        plt.plot(self._data_per_gen.T[0], self._data_per_gen.T[2], '-', label='Fitness médio')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.plot(self._data_per_gen.T[0], self._data_per_gen.T[1], '-', label='Fitness minimo')
        plt.plot(self._data_per_gen.T[0], self._data_per_gen.T[3], '-', label='Fitness máximo')
        plt.legend(loc="upper left")
        plt.title('Evolução do fitness máximo, médio e minimo')
        plt.subplot(111).legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    @staticmethod
    def init_population(axis_range, n_pop, dimensions):
        return uniform(*axis_range, size=(n_pop, dimensions))

    @staticmethod
    def pop_fitness(pop, func):
        return np.array(list(map(func, pop)))

    @staticmethod
    def pick_rs(pop):
        index = choice(len(pop), 3, replace=False)
        return deepcopy(pop[index])

    @staticmethod
    def select_new_generation(pop, children, func):
        def pick_child_or_father(funct, child, father):
            if funct(child) <= funct(father):
                return child
            else:
                return father

        new_gen = [pick_child_or_father(func, child, father) for father, child in zip(pop, children)]
        return np.array(new_gen)

    @staticmethod
    def gen_children(pop, C=None, F=None):
        def gen_selector():
            if C is None:
                return uniform(0.6, 0.9)
            else:
                return C

        def gen_scale():
            if F is None:
                return uniform(0.7, 0.9)
            else:
                return F

        def pick_coordinates(being, rs, index, delta):
            if uniform() <= gen_selector() or delta == index:
                return rs[0][index] + gen_scale() * (rs[1][index] - rs[2][index])
            else:
                return being[index]

        def gen_child(being):
            rs = DifferentialEvolution.pick_rs(pop)
            delta = randint(0, rs.shape[1])
            return np.array(
                [
                    pick_coordinates(being, rs, index, delta) for index, value in enumerate(being)
                ]
            )

        return np.array([gen_child(being) for being in pop])


def main():
    n_pop = 100
    axis_ranges = (-2, 2)
    func = rastrigin

    de = DifferentialEvolution(
        n_pop,
        axis_ranges,
        max_gen_to_converge=20
    )
    de.arg_min(func, True)
    de.plot()

if __name__ == '__main__':
    main()
