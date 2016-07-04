__author__ = 'vinay_vijayan'

# TODO clean up code, make it OO
# TODO why isn't distrbuted processing not working if OO?
# TODO check Scoop documentation ?
# TODO tune GA parameters


import warnings
warnings.filterwarnings("ignore")

from scoop import futures

import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from entity_facility import EntityFacility


GRID_SIZE = 5
IND_INIT_SIZE = GRID_SIZE * GRID_SIZE

NGEN = 3
MU = 50
LAMBDA = 4
CXPB = 0.7
MUTPB = 0.2

random.seed(64)

list_entity_object = []
for i in range(0, GRID_SIZE):
    list_dummy = []
    for j in range(0, GRID_SIZE):
        list_dummy.append(EntityFacility())
    list_entity_object.append(list_dummy)

creator.create("Fitness", base.Fitness, weights=(-0.5, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_usage_cost(individual):
    array_reshaped = np.reshape(individual, (GRID_SIZE, GRID_SIZE))
    total_served = 0
    total_cost = 0
    for i in range(0, GRID_SIZE):
        for j in range(0, GRID_SIZE):
            if i - 1 > 0:
                north_start_time = array_reshaped[i-1][j]
            else:
                north_start_time = 10
            if i + 1 < GRID_SIZE:
                south_start_time = array_reshaped[i+1][j]
            else:
                south_start_time = 10
            if j+1 < GRID_SIZE:
                east_start_time = array_reshaped[i][j+1]
            else:
                east_start_time = 10
            if j-1 >= 0:
                west_start_time = array_reshaped[i][j-1]
            else:
                west_start_time = 10
            centre_start_time = array_reshaped[i][j]
            list_entity_object[i][j].simulate(dict_start_times={'centre': centre_start_time, 'north': north_start_time,
                                                                'south': south_start_time, 'east': east_start_time,
                                                                'west': west_start_time})
            people_served, installation_cost = list_entity_object[i][j].get_stats()
            total_served += people_served
            total_cost += installation_cost

    return total_cost, total_served

def mate_individuals(individual_1, individual_2):
    half_length = len(individual_1)/2

    individual_1_first_half = individual_1[:half_length]
    individual_1_second_half = individual_1[half_length:]

    individual_2_first_half = individual_2[:half_length]
    individual_2_second_half = individual_2[half_length:]

    child_1 = individual_1_first_half + individual_2_second_half
    child_2 = individual_2_first_half + individual_1_second_half

    return creator.Individual(child_1), creator.Individual(child_2)

def mutate_individual(individual):
    index = random.randrange(0, len(individual))
    individual[index] = 0
    return individual,


def main():

    toolbox.register("evaluate", eval_usage_cost)
    toolbox.register("mate", mate_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("map", futures.map)

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()