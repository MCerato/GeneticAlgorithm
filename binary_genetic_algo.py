# *****traitement des accents dans le texte****************
# -*- coding: utf-8 -*-

# =============================================================================
import random
import math
import matplotlib.pyplot as plt
# =============================================================================


# In[0]: Descritption

# =============================================================================
# class Thread_Command(threading.Thread):
#     """
#     Description
#     ----------
#     Trying to find the maximum of a binary word using genetic algorithm
#
#     Information
#     ----------
#     - Last_Modification_date = 29/08/2019
#     - version : 1.0.0.0.0
#     - author : M.C
#
#     Note
#     ----------
#
#     Parameters
#     ----------
#     first : Name
#       object resulting of ...
#
#     second : Name
#       object resulting of ...
#     """
# =============================================================================


# In[1]: population intialization
def InitialPopulation(populationSize = 100, chromosomeSize = 10):
    population = []

    for i in range(populationSize):
        population.append(bin(random.getrandbits(chromosomeSize))) # create a random pool of chromosomes (1st generation)
        population[i] = population[i][2:] # takes off the "0b" from the binary number generated in string
        population[i] = population[i].rjust(chromosomeSize, '0')# display any generated number on 10bits (add 0 on the left of generated numbers)

    return population


# In[2]: fitness function
def Fitness(population):
    fitnessValues = []
    globalFitness = 0

    for i, chromosome in enumerate(population):
        fitnessValues.append(chromosome.count('1'))# count the number of "1" in the population and put the result in a list

    bestValue = max(fitnessValues)
    fittestIndex = fitnessValues.index(bestValue)
    fittestValue = population[fittestIndex]
    return fittestValue, fitnessValues

# In[3]: condition of selection
def FitnessProbability(fitnessValues):
    fitnessProbability = []

    globalFitness = sum(fitnessValues)# calculate the global value of fitness
    for i, chromosomeFitness in enumerate(fitnessValues):

# /!\=================Only positive result==================================/!\
        fitnessProbability.append(chromosomeFitness / globalFitness) # probability calculated by currentfitness/ sum(allfitnesses)
        # print(f"Chromosome Number : {i} Chromosome : {population[i]} fitness value {fitnessValues[i]} fitness proba {fitnessProbability[i]} ")
# =============================================================================
    return fitnessProbability

# In[3]: selection of the fittest
def FittestSelection(population, fitnessProbability, populationSize):
    fittest = []
    fittest = random.choices(population, weights = fitnessProbability, k = math.ceil(populationSize/2)) # selct the fittest randomly based on their fitness value (weighted).
    return fittest


# In[4]: mate selector
def MateCrossover1Point(fittest):

    randomParents = []
    offprings = []
    newGeneration = []

    for i in range(math.ceil(populationSize/4)):
        randomParents.append(random.sample(fittest, 2)) # Random couple selection in the fittest group (no weight)

    for mates in randomParents:
        mom = mates[0]
        dad = mates[1]

        x = int(random.uniform(0, len(mom)))# choose a random index to crossover chromosomes

        offprings1 = mom[:x] + dad[x:] # generate offspring1 with mom + dad around crossover point
        newGeneration.append(offprings1) # add offspring1 the the new generation population
        offprings2 = dad[:x] + mom[x:] # generate offspring1 with dad + mom around crossover point
        newGeneration.append(offprings2)# add offspring2 the the new generation population

    return newGeneration

def MateCrossoverUniform(fittest):

    randomParents = []
    newGeneration = []

    for i in range(math.ceil(populationSize/4)):
        randomParents.append(random.sample(fittest, 2))  # Random couple selection in the fittest group (no weight)

    for mates in randomParents:
        mom = mates[0]
        dad = mates[1]
        offprings1 =""
        offprings2 =""

        for i in range(len(mom)):
            flipCoin = random.getrandbits(1) #  # generates randomly a "1" or a "0" (flip a coin)

            if bool(flipCoin) is True:

                offprings1 = offprings1 + mom[i] # offprings1 is using mom gene
                offprings2 = offprings2 + dad[i] # offprings2 is using dad gene
            else:
                offprings1 = offprings1 + dad[i] # offprings1 is using dad gene
                offprings2 = offprings2 + mom[i] # offprings2 is using mom gene

        newGeneration.append(offprings1) # add offspring1 the the new generation population
        newGeneration.append(offprings2) # add offspring2 the the new generation population
        del offprings1 # delete the immutable
        del offprings2 # delete the immutable

    return newGeneration


# In[5]: random mutation
def Mutation(oldGeneration, offpringsGeneration, mutationRate, populationSize, chromosomeSize):

    nbOfMutation = math.ceil((populationSize)*chromosomeSize*mutationRate)  # Nb of mutation in the next generation
    oldGeneration.extend(offpringsGeneration)  # mixing the fittest generation and their offsprings
    newPopulation = oldGeneration.copy()  # rename of Population
    del oldGeneration

    for i in range(nbOfMutation):
        indexOfChromosomeToMutate = int(random.uniform(0, populationSize-1)) # getting the index of the chromosome to mutate in the population
        chromosomeToMutate = newPopulation[indexOfChromosomeToMutate] # getting wich chromosome to mutate
        indexOfGeneToMutate = int(random.uniform(0, chromosomeSize-1)) # getting the index of the gene to mutate in the chromosome
        geneToMutate = chromosomeToMutate[indexOfGeneToMutate] # getting the gene to mutate in the chromosome

        if geneToMutate == "1":
        # flip the chosen gene
           mutatedChromosome = chromosomeToMutate[:indexOfGeneToMutate]+"0"+chromosomeToMutate[indexOfGeneToMutate+1:]

        else:
            mutatedChromosome = chromosomeToMutate[:indexOfGeneToMutate]+"1"+chromosomeToMutate[indexOfGeneToMutate+1:]

        newPopulation[indexOfChromosomeToMutate] = mutatedChromosome

    return newPopulation


# In[6]: success criteria
def SuccessCriteria(population, populationSize, chromosomeSize):
    successPopulation = 0
    succesRatios = 0

    for chromosome in population:
        successPopulation += chromosome.count('1')
    successRatio = (100*successPopulation)/(populationSize*chromosomeSize)


# ??????????
    if successPopulation == populationSize*chromosomeSize:
        return {'highestFitness': successPopulation, 'successRatio': successRatio}
    else:
        return {'highestFitness': successPopulation, 'successRatio': successRatio}
# ??????????

# In[7]: plotting results
def PlottingOfResult(successPopulation, successRatios, fittestValue):
    plt.figure(figsize=(14, 6))
    XAxis = [val for val in range(0, len(successRatios))]



    plt.subplot(1, 2, 1) # 1 line, 2 col, subfigure 1
    plt.title("Value of fittest (the best is the size of the chromosome)")
    plt.plot(XAxis, fittestValue, c="red", lw=1, label="Value of fittest (best is size of the chromosome)")
    plt.ylabel("best result chromosome")
    plt.xlabel("Nb of generations")

    plt.subplot(1, 2, 2) # 1 line, 2 col, subfigure 1
    plt.title("all '1' chromosomes in population would be 100%")
    plt.plot(XAxis, successRatios, c="blue", lw=1, label="matching ratio")
    plt.ylabel("%")
    plt.xlabel("Nb of generations")

    plt.legend()
    plt.show()


# In[8]: Main Program

# ******* Jeux de paramètres *********
maxGeneration = 500
populationSize = 8000
chromosomeSize = 80
mutationRate = 0.001
# ************************************

SuccessRatios = []
successChromosomeValue = []
successPopulationValue = []

population = InitialPopulation(populationSize, chromosomeSize)
print(f" Initial population {population}")
for i in range(maxGeneration):

    fittestValue, fitnessValues = Fitness(population)
# =============================================================================
#     print(f"")
#     print(f"best Value : {fittestValue}")
# =============================================================================

    fitnessProbability = FitnessProbability(fitnessValues)
# =============================================================================
#     print(f"")
#     print(f"probability table : {fitnessProbability}")
# =============================================================================

    fittest = FittestSelection(population, fitnessProbability, populationSize)
# =============================================================================
#     print(f"")
#     print(f"chromosomes chosen : {fittest}")
# =============================================================================

    offpringsGeneration = MateCrossoverUniform(fittest)
# =============================================================================
#     print(f"")
#     print(f"offsprings Generation : {offpringsGeneration}")
# =============================================================================

    population = Mutation(fittest, offpringsGeneration, mutationRate, populationSize, chromosomeSize)
# =============================================================================
#     print(f"")
#     print(f"New Population : {population}")
# =============================================================================

# ---------------------
    result = SuccessCriteria(population, populationSize, chromosomeSize)
    SuccessRatios.append(result['successRatio'])


    successPopulationValue.append(result['highestFitness'])
    successChromosomeValue.append(fittestValue.count('1'))

print (f"Final population : {population}")
print(f"{max(SuccessRatios)}% of all bits are '1' ")
print(f"Best Value of this run : {max(successChromosomeValue)}")

PlottingOfResult(successPopulationValue, SuccessRatios, successChromosomeValue)