import math
import numpy as np
from deap import base
from deap import creator
from deap import tools


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
        smin, smax = self.dispatc_speed()

        if self.minimization is True:
            creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        else:
            creator.create("FitnessMax", base.Fitness, weights=(-1.0,))

        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=list,
                       smin=None, smax=None, best=None)
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

    def dispatc_speed(self):
        return self.SPEED[0], self.SPEED[1]

    def dispatch_boundaries(self):
        return self.BOUNDARY[0], self.BOUNDARY[1]


settings = {"type": "PSO",
            "ac_type": "EI",
            "n_search": 5,
            "boundaries": [[0, 1] for i in range(6)],
            "epsilon": 0.01,
            "iteration": 2,
            "minimization": True,
            "optimization": True,
            "n_restart": 5,
            "sampling": np.random.uniform}
