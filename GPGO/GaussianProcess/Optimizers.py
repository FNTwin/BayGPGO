import math
import array
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import diversity, convergence, hypervolume


class PSO():

    def __init__(self, dim, boundary, population=5, gen=1000, minimization=True, func=None):
        # POTREI AGGIUNGERE SPEED
        self.DIM = dim
        self.BOUNDARY = boundary
        self.POPULATION = population
        self.GEN = gen
        self.SPEED = self.initialize_speed(boundary)
        self.func = func
        self.minimization = minimization
        self.FIRST = True

    def generate(self, size, pmin, pmax, smin, smax):
        # pmin,pmax=self.dispatch_boundaries()
        part = creator.Particle(np.random.uniform(pmin, pmax, size))
        part.speed = np.random.uniform(smin, smax, size)
        part.smin = smin
        part.smax = smax
        return part

    def updateParticle(self, part, best, phi1, phi2):
        u1 = np.random.uniform(0, phi1, len(part))
        u2 = np.random.uniform(0, phi2, len(part))
        v_u1 = u1 * (part.best - part)
        v_u2 = u2 * (best - part)
        part.speed += v_u1 + v_u2
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part += part.speed

    def initialize_speed(self, boundaries):
        s_b = np.sort(boundaries.sum(axis=1) / 2)
        return [-s_b[math.floor(len(s_b) / 2)], s_b[math.floor(len(s_b) / 2)]]

    def run(self):
        # Setup with dummy variables
        n = self.GEN
        dim = self.DIM
        population = self.POPULATION
        # pmin,pmax=self.dispatch_boundaries()
        smin, smax = self.dispatch_speed()

        if self.FIRST:
            if self.minimization is True:
                creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
            else:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list,
                           smin=None, smax=None, best=None)
            self.FIRST = False

        toolbox = base.Toolbox()
        toolbox.register("particle", self.generate, size=dim, pmin=self.BOUNDARY[:, 0],
                         pmax=self.BOUNDARY[:, 1], smin=smin, smax=smax)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", self.updateParticle, phi1=2.0, phi2=2.0)
        toolbox.register("evaluate", self.func)

        pop = toolbox.population(n=population)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = n
        best = None

        for g in range(GEN):
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values

            for part in pop:
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            # print(logbook.stream)

        self.PSO = {
            "pop": pop,
            "logbook": logbook,
            "best": best
        }

        return best

    def dispatch_speed(self):
        return self.SPEED[0], self.SPEED[1]

    def dispatch_boundaries(self):
        return self.BOUNDARY[0], self.BOUNDARY[1]

    def pop_sort(self):
        return self.PSO["pop"].sort(key=lambda x: x.fitness.values)


class NSGAII():

    def __init__(self, dim, nobj, boundary, population=100, gen=200, minimization=True, CXPB=0.9, func=None):

        self.DIM = dim
        self.NOBJ= nobj
        self.BOUNDARIES = boundary
        self.POPULATION = population
        self.GEN = gen
        self.func = func
        self.minimization = minimization
        self.CXPB = CXPB
        self.FIRST = True

    def uniform(self, b):
        return np.random.uniform(b[:, 0], b[:, 1], self.DIM)

    def run(self):
        n = self.GEN
        dim = self.DIM
        population = self.POPULATION


        if self.FIRST:
            if self.minimization is True:
                creator.create("FitnessMin", base.Fitness, weights=tuple([-1.0]* self.NOBJ))
                creator.create("Individual", np.ndarray, typecode='d',
                               fitness=creator.FitnessMin)
            else:
                creator.create("FitnessMax", base.Fitness, weights=tuple([1.0] * self.NOBJ))
                creator.create("Individual", np.ndarray, typecode='d',
                               fitness=creator.FitnessMax)
                #array.array
            self.FIRST = False

        toolbox = base.Toolbox()
        toolbox.register("attr_float", self.uniform, self.BOUNDARIES)
        toolbox.register("individual",
                         tools.initIterate,
                         creator.Individual,
                         toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.func)
        toolbox.register("mate",
                         tools.cxSimulatedBinaryBounded,
                         low=self.BOUNDARIES[:, 0].tolist(),
                         up=self.BOUNDARIES[:, 1].tolist(),
                         eta=20.0)
        toolbox.register("mutate",
                         tools.mutPolynomialBounded,
                         low=self.BOUNDARIES[:, 0].tolist(),
                         up=self.BOUNDARIES[:, 1].tolist(),
                         eta=20.0,
                         indpb=1.0 / dim)

        toolbox.register("select", tools.selNSGA2)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)

        # Begin the generational process
        for gen in range(1, n):

            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.CXPB:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, population)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            # print(logbook.stream)

        front = np.array([ind.fitness.values for ind in pop])
        pop_array=np.array([ind for ind in pop])
        hyper_v=hypervolume(pop)

        self.NSGAII = {
            "pop": pop,
            "logbook": logbook,
            "pareto": front,
            "npop" : pop_array,
            "hypervolume" : hyper_v
        }

        return pop, logbook, front

    def pareto_sorted(self):
        if hasattr(self, 'NSGAII'):
            tmp_array=self.NSGAII["pareto"]
            return tmp_array[tmp_array[:, 0].argsort()]
        else:
            print("NO PARETO SOLUTION IN MEMORY")
