import os
import random
import pickle
import time
import multiprocessing

import numpy as np
from deap import algorithms, base, creator, tools

from CADETProcess.common import settings
from CADETProcess.dataStructure import UnsignedInteger, UnsignedFloat
from CADETProcess.optimization import SolverBase, OptimizationResults

class DEAP(SolverBase):
    """ Adapter for optimization with an Genetic Algorithm called DEAP.

    Defines the solver options, the statistics, the history, the logbook and
    the toolbox for recording the optimization progess. It implements the
    abstract run method for running the optimization with DEAP.

    Attributes
    ----------
    optimizationProblem: optimizationProblem
        Given optimization problem to be solved.
    options : dict
        Solver options, default set to None, if nothing is given.

    See also
    --------
    base
    tools
    Statistics
    """
    cxpb = UnsignedFloat(default=1)
    mutpb = UnsignedFloat(default=1)
    sig_figures = UnsignedInteger(default=3)
    seed = UnsignedInteger(default=12345)
    _options = ['cxpb', 'mutpb', 'sig_figures', 'seed']

    def run(self, optimization_problem, n_gen=None, population_size=None,
            use_multicore=True, use_checkpoint=True):

        self.optimization_problem = optimization_problem
        # Abbreviations
        lb = optimization_problem.lower_bounds
        ub = optimization_problem.upper_bounds
        n_vars = optimization_problem.n_variables

        # Settings
        if population_size is None:
            population_size = min(200, max(25*len(optimization_problem.variables),50))
        if n_gen is None:
            n_gen = min(100, max(10*len(optimization_problem.variables),40))

        # NSGA3 Settings
        n_obj = 1
        p = 4
        ref_points = tools.uniform_reference_points(n_obj, p)

        # !!! emo functions breaks if n_obj == 1, this is a temporary fix
        if n_obj == 1:
            def sortNDHelperB(best, worst, obj, front):
                if obj < 0:
                    return
                sortNDHelperB(best, worst, obj, front)

            tools.emo.sortNDHelperB = sortNDHelperB

        # Definition of classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Tools
        toolbox = base.Toolbox()

        # Map for parallel evaluation
        manager = multiprocessing.Manager()
        cache = manager.dict()
        pool = multiprocessing.Pool()
        if use_multicore:
            toolbox.register("map", pool.map)

        # Functions for creating individuals and population
        toolbox.register(
            "individual",
            tools.initIterate, creator.Individual,
            optimization_problem.create_initial_values
        )
        def initIndividual(icls, content):
            return icls(content)
        toolbox.register("individual_guess", initIndividual, creator.Individual)
        
        def initPopulation(pcls, ind_init, population_size):
            population = optimization_problem.create_initial_values(population_size)
            return pcls(ind_init(c) for c in population)
        
        toolbox.register(
            "population", initPopulation, list, toolbox.individual_guess, 
        )

        # Functions for evolution
        toolbox.register("evaluate", self.evaluate, cache=cache)
        toolbox.register(
            "mate", tools.cxSimulatedBinaryBounded, low=lb, up=ub, eta=30.0
        )
        toolbox.register(
            "mutate", tools.mutPolynomialBounded,
             low=lb, up=ub, eta=20.0, indpb=1.0/n_vars
        )
        toolbox.register(
            "select", tools.selNSGA3, nd="standard", ref_points=ref_points
        )

        # Round individuals to prevent reevaluation of similar individuals
        def round_individuals():
            def decorator(func):
                def wrapper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        for index, el in enumerate(child):
                            child[index] = round(el, self.sig_figures)
                    return offspring
                return wrapper
            return decorator
        toolbox.decorate("mate", round_individuals())
        toolbox.decorate("mutate", round_individuals())


        statistics = tools.Statistics(key=lambda ind: ind.fitness.values)
        statistics.register("min", np.min)
        statistics.register("max", np.max)
        statistics.register("avg", np.mean)
        statistics.register("std", np.std)

        # Load checkpoint if present
        checkpoint_path = os.path.join(
            settings.project_directory,
            optimization_problem.name + '/checkpoint.pkl'
        )

        if use_checkpoint and os.path.isfile(checkpoint_path):
            # A file name has been given, then load the data from the file
            with open(checkpoint_path, "rb") as cp_file:
                cp = pickle.load(cp_file)

            self.population = cp["population"]
            start_gen = cp["generation"]
            self.halloffame = cp["halloffame"]
            self.logbook = cp["logbook"]
            random.setstate(cp["rndstate"])
        else:
            # Start a new evolution
            start_gen = 0
            self.halloffame = tools.HallOfFame(maxsize=1)
            self.logbook = tools.Logbook()
            self.logbook.header = "gen", "evals", "std", "min", "avg", "max"

            # Initialize random population
            random.seed(self.seed)
            self.population = toolbox.population(population_size)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [
                ind for ind in self.population if not ind.fitness.valid
            ]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Compile statistics about the population
            record = statistics.compile(self.population)
            self.logbook.record(gen=0, evals=len(invalid_ind), **record)

        # Begin the generational process
        start = time.time()
        for gen in range(start_gen, n_gen):
            self.offspring = algorithms.varAnd(
                self.population, toolbox, self.cxpb, self.mutpb
            )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            self.population = toolbox.select(
                self.population + self.offspring, population_size
            )

            # Compile statistics about the new population
            record = statistics.compile(self.population)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            self.halloffame.update(self.population)

            # Create Checkpoint file
            cp = dict(
                population=self.population, generation=gen,
                halloffame=self.halloffame, logbook=self.logbook,
                rndstate=random.getstate()
            )

            with open(checkpoint_path, "wb") as cp_file:
                pickle.dump(cp, cp_file)

            best = self.halloffame.items[0]
            self.logger.info(
                'Generation {}: x: {}, f: {}'.format(
                    str(gen), str(best), str(best.fitness.values[0])
                )
            )

        elapsed = time.time() - start

        x = self.halloffame.items[0]

        eval_object = optimization_problem.set_variables(x, make_copy=True)
        if self.optimization_problem.evaluator is not None:
            frac = optimization_problem.evaluator.evaluate(
                eval_object, return_frac=True
            )
            performance = frac.performance
        else:
            frac = None
            performance = optimization_problem.evaluate(x, force=True)
        f = optimization_problem.objective_fun(performance)

        results = OptimizationResults(
            optimization_problem = optimization_problem,
            evaluation_object = eval_object,
            solver_name = str(self),
            solver_parameters = self.options,
            exit_flag = 1,
            exit_message = 'DEAP terminated successfully',
            time_elapsed = elapsed,
            x = list(x),
            f = f,
            c = None,
            frac = frac,
            performance = performance.to_dict()
        )

        return results


    def evaluate(self, ind, cache=None):
        results = self.optimization_problem.evaluate_objectives(
            ind, make_copy=True, cache=cache
        )
        return results
