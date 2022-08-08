# *****traitement des accents dans le texte****************
# -*- coding: utf-8 -*-

# =============================================================================
import random
import math
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================


# In[0]: Descritption

# =============================================================================
# class Name(dependencies):
#     """
#     Description
#     ----------
#     Solving a continuous equation
#     this equation accept 2 float parameters : x and y
#
#     Information
#     ----------
#     - Last_Modification_date = 11/02/2022
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

# In[1]: setup the environment test (if necessary)
def setupFunctionLimitation(minValue, maxValue):
    """
    Description
    ----------
    Set the limitations of the system.

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    minValue (signed int)
      minimum value that can be taken by the parameters

    maxValue (signed int)
      maximum value that can be taken by the parameters

    Return
    ----------
    minValue (signed int)
      minimum value that can be taken by the parameters

    maxValue (signed int)
      maximum value that can be taken by the parameters

    FitnessFunction (funtion)
      Function to evaluate can be defined here (/!\ numpy format /!\)

    """
# ========================fitness function=====================================
    def FitnessFunction(x, y):
        z = x*np.sin(4*x) + 1.1*y*np.sin(2*y)
        return z
# =============================================================================

    return minValue, maxValue, FitnessFunction


# In[2]: population intialization
def InitialPopulation(populationSize = 20, chromosomeSize = 2, minValue = -10, maxValue = 10):
    """
    Description
    ----------
    Create a random pool of chromosomes (1st generation) composed of 2
     parameters - coordinates [x, y]

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    populationSize (unsingned int)
      Define the number of coordinates [x, y] in the population (20 by default)

    chromosomeSize (unsingned int)
      Define the number of parameters (2 for x and y)

    minValue (singned int)
      minimum value that can be taken by the parameters

    maxValue (singned int)
      maximum value that can be taken by the parameters

    Return
    ----------
    population (list)
      A list of "PopulationSize" number of float coordinates [x,y] chosen
      in between min and max limitation
      ex : [[3.456, 7.654], [9.876, 1.123]]

    """
    population = []

    for chromosomeIndex in range(populationSize):
        population.append([random.uniform(minValue, maxValue) for i in range(chromosomeSize)]) # create a random pool of chromosomes (1st generation

    return population

# In[3]: fitness function(s)
def Fitness(population, fitnessFunction):
    """
    Description
    ----------
    Pass the population of [x, y] trhough the fitness function and return the
    results in list

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    population (list)
      list of chromosomes (here those are a number of [x, y])

    fitnessFunction (function)
      The dual variables mathematical function to be used

    return
    ----------
    fitnessValues (list)
      A list of results from the population passed through the fitness funtion

    """
    fitnessValues = []

    for chromosome in population:
        fitnessValues.append(fitnessFunction(chromosome[0], chromosome[1]))# put population in fitness function and put the result in a list

    return fitnessValues

# In[3]: condition of selection
def GoalCondition(targetValue):
    """
    Description
    ----------
    control if the goal is coherent and make it easier to use

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    targetValue (string, int, float)
      Value (or extremum) looked for in function. it can either be :
      - the word "maximum"
      - the word "minimum"
      - a number

    return
    ----------
    targetValue (None, float, int)
      A list of results from the population passed through the fitness funtion

    """
    if targetValue == "maximum":
        return max(fitnessValues)

    elif targetValue == "minimum":
        return min(fitnessValues)

    elif type(targetValue) == str and (targetValue != "maximum", "minimum"):
        print(f"")
        print(f"wrong Value. Target not clear enough")
        return None

    else:
        return targetValue

# In[3]: condition of selection
def FitnessProbability(fitnessValues, population, targetValue = "maximum"):
    """
    Description
    ----------
    Define a table of probability according to the "distance" between a target
    value and the result of the fitness function

    For instance if target is 0, the closest value of 0 will have the highest
    probability (weight), the second closest will have the second highest
    probability, etc...

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    fitnessValues (list)
      List of fitness values (previously calculated)

    population (list)
      List of chromosomes (here those are a number of [x, y])

    targetValue  (string, int, float)
      Optimal value looked for

    Return
    ----------
    fitnessProbability (list)
      Give a llist of weight (or probability) according to the fitness values
      and target given (closest values to the target have higher probability)

    fittestChromosome (list)
      Extract the best couple of values [x, y] for this generation
      (for visualization only)

    fittestValue (float, int)
        Extract the best result of the fitness funtion for this generation
        (for visualization only)

    """
    fitnessOrder = []
    fitnessProbability = []
    rankTable = [None for i in range(len(fitnessValues))]

# =============================================================================
# Create a list of "distance" between target and value
    for val in fitnessValues:
        fitnessOrder.append(abs(targetValue - val))# get the "distance" between target and value checked
# =============================================================================

    tempFitnessOrderValue = fitnessOrder.copy()# create a value list of fitness order
    tempFitnessOrderIndex = fitnessOrder.copy()# create an index list of fitness order
    rankValue = len(fitnessOrder) # the rank value start at the maximum

# =============================================================================
# while it remains at least one value in the "temp fitness order"
    while tempFitnessOrderValue:

# get the closest value from the target
        valMini = min(tempFitnessOrderValue)

# check if a value is repeated and give the same rank value if it is the case
        nbValMini = tempFitnessOrderValue.count(valMini)

        while nbValMini:
            fitnessOrderIndex = tempFitnessOrderIndex.index(valMini)
            rankTable[fitnessOrderIndex] = rankValue # give the rank value

            tempFitnessOrderIndex[fitnessOrderIndex] = None # value is taken off of the index table
            tempFitnessOrderValue.pop(tempFitnessOrderValue.index(valMini)) # index and value are taken off of the value table
            nbValMini -= 1

        rankValue -= 1
# =============================================================================

# =============================================================================
# calculate the probability of each value to be selected and put it in list
    for rank in rankTable:
        fitnessProbability.append(rank/sum(rankTable))
# =============================================================================

# =============================================================================
# calculate the fittest chromosome and the fittest value of this generation
    fittestIndex = rankTable.index(max(rankTable)) # find the table index
    fittestChromosome = population[fittestIndex] # retrieves the best chromosome in population according to the index
    fittestValue = fitnessValues[fittestIndex] # retrieves the best value found after calculation
# =============================================================================

    del tempFitnessOrderIndex
    del tempFitnessOrderValue

    return fitnessProbability, fittestChromosome, fittestValue

# In[4]: selection of the fittest
def FittestSelection(population, fitnessProbability, populationSize):
    """
    Description
    ----------
    Select the part of the population which fits the best

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------
    This selection is deterministic : The best half (the one with the best
    probability) of the population is always selected

    Parameters
    ----------
    population (list)
      Value (or extremum) looked for in function. it can either be :
      - the word "maximum"
      - the word "minimum"
      - a number

    fitnessProbability (list)
      Give a list of weight (or probability) according to the fitness values
      and target given (closest values to the target have higher probability)

    populationSize (unsingned int)
      Define the number of coordinates [x, y] in the population (20 by default)

    return
    ----------
    fittest (list)
      hHe list of the upper half fittest chromosomes in the population

    """
    fittest = []
    tempFitnessProbability = fitnessProbability.copy()

# =================Keep the fittest chromosomes ===============================
    for i in range(math.ceil(populationSize/2)):
        maxProba = max(tempFitnessProbability)
        probaIndex = fitnessProbability.index(maxProba)
        fittest.append(population[probaIndex])
        tempFitnessProbability.pop(tempFitnessProbability.index(maxProba))
# =============================================================================

    return fittest

# In[5]: mate selection
def MateSelection(fittest, populationSize):
    """
    Description
    ----------
    Create couples of values [x, y] to create "parents" before create next
    generation
    Parent A : [x1, y1]
    Parent B : [x2, y2]
    parents : [[x1, y1], [x2, y2]]

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------
    Sample is done without replacement wich means that a value [x, y] picked
    as "mom", can't be picked as "dad" at the same time BUT...
    it can be selected to be an other "mom" or "dad" in next samples

    Parameters
    ----------
    fittest (list)
      The list of the upper half fittest chromosomes in the population

    populationSize (unsingned int)
      Define the number of coordinates [x, y] in the population (20 by default)

    return
    ----------
    randomParents (list)
      A list of formed couples randomly selected

    """
    randomParents = []

    for i in range(math.ceil(populationSize/4)):
        randomCouple = random.sample(fittest, 2)
        randomParents.append(randomCouple) # Random couple selection in the fittest group (no weight)

    return randomParents

# In[6]: mating of chromosomes
def MateHauptMethod(randomParents):
    """
    Description
    ----------
    create 2 offsprings based on 2 parents through the Haupt&Haupt method

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    randomParents (list)
      A list of formed couples randomly selected

    populationSize (unsingned int)
      Define the number of coordinates [x, y] in the population (20 by default)

    return
    ----------
    newGeneration (list)
      A list of the offsprings generated (without the old generation)
    """
    newGeneration = []

    for mates in randomParents:
        mom = mates[0]
        dad = mates[1]
        beta = random.random()

        flipCoin = random.getrandbits(1) #  # generates randomly a "1" or a "0" (flip a coin)

# =============================================================================
#         print(f"")
#         print(f"mom : {mom} | dad : {dad}")
#         print(f"Flip coin : {flipCoin} | beta : {beta}")
# =============================================================================

        if bool(flipCoin) is True:
# y is the parameter passed to offsprings trough Haupt Method
            offprings1 = [mom[0], (1-beta)*mom[1] + beta*dad[1]]
            offprings2 = [dad[0], (1-beta)*dad[1] + beta*mom[1]]

        else:
# x is the parameter passed to offsprings trough Haupt Method
            offprings1 = [(1-beta)*mom[0] + beta*dad[0], mom[1]]
            offprings2 = [(1-beta)*dad[0] + beta*mom[0], dad[1]]

        newGeneration.append(offprings1)
        newGeneration.append(offprings2)

# =============================================================================
#         print(f"")
#         print(f"offprings1 : {offprings1}")
#         print(f"offprings2 : {offprings2}")
# =============================================================================
    return newGeneration

# In[7]: random mutation
def Mutation(oldGeneration, offpringsGeneration, mutationRate, populationSize, chromosomeSize, minValue, maxValue):
    """
    Description
    ----------
    create 2 offsprings based on 2 parents through the Haupt&Haupt method

    Information
    ----------
    - Last_Modification_date = 11/02/2022
    - version : 1.0.0.0.0
    - author : M.C

    Note
    ----------

    Parameters
    ----------
    oldGeneration (list)
      The fittest of previous generation of chromosome

    offpringsGeneration (list)
      offsprings previously genreted through crossover

    mutationRate (float)
      Define the rate of mutation in the population

    populationSize (unsingned int)
      Define the number of coordinates [x, y] in the population (20 by default)

    chromosomeSize (unsingned int)
      Define the number of parameters (2 for x and y)

    minValue (signed int)
      minimum value that can be taken by the parameters

    maxValue (signed int)
      maximum value that can be taken by the parameters

    return
    ----------
    newPopulation (list)
      merge of the fittest of old generation and the mutated offsprings
    """
    # the objective is to change randomly a value in betxeen limits
    mutatedChromosome =[]

    nbOfMutation = math.ceil((populationSize)*chromosomeSize*mutationRate)  # Nb of mutation in the next generation

    for i in range(nbOfMutation):
        indexOfChromosomeToMutate = int(random.uniform(0, len(offpringsGeneration))) # getting the index of the chromosome to mutate in the population
        chromosomeToMutate = offpringsGeneration[indexOfChromosomeToMutate] # getting what chromosome to mutate according to the index
        indexOfGeneToMutate = int(random.uniform(0, chromosomeSize)) # getting the index of the gene to mutate in the chromosome

        mutatedChromosome = chromosomeToMutate.copy()
        mutatedChromosome[indexOfGeneToMutate] = random.uniform(minValue, maxValue)
        offpringsGeneration[indexOfChromosomeToMutate] = mutatedChromosome

    oldGeneration.extend(offpringsGeneration)
    newPopulation = oldGeneration.copy()

    return newPopulation

# In[8]: success criteria

# In[9]: plotting results
def PlottingOfResult(maxGeneration, targetValue, successPopulationValue, successChromosomeValue, func, minValue, maxValue):
    elevation = []
    distanceOfTarget = []
    nbElevation = 200

    xs = []
    ys = []

    for value in successPopulationValue:
        distanceOfTarget.append(abs(targetValue - value)) # get the "distance" between target and value checked

    bestValue = successPopulationValue[distanceOfTarget.index(min(distanceOfTarget))]
    bestChromosome = successChromosomeValue[distanceOfTarget.index(min(distanceOfTarget))]

    print (f"The target value : {targetValue}")
    print (f"closest altitude of target value : {bestValue}")
    print (f"closest coordinate of target value : {bestChromosome}")

    plt.figure(figsize=(18, 8))

    XAxisValue = [val for val in range(0, maxGeneration)]
    xCoord, yCoord = np.meshgrid(np.linspace(minValue, maxValue, 200),
                                 np.linspace(minValue, maxValue, 200))

    elevation = func(xCoord, yCoord)
    levels = np.linspace(elevation.min(), elevation.max(), nbElevation)

    plt.subplot(1, 2, 1) # 1 line, 2 col, subfigure 1
    plt.title("Value of the best altitude")
    plt.plot(XAxisValue, successPopulationValue, lw=1, label="plop")
    plt.ylabel("elevation value")
    plt.xlabel("Nb of generations")

    plt.subplot(1, 2, 2) # 1 line, 2 col, subfigure 2
    plt.title("map topography")
    plt.ylabel("coordinates Y")
    plt.xlabel("coordinates X")

    plt.contourf(xCoord, yCoord, elevation, levels = levels)

    for value in successChromosomeValue:
       xs.append(value[0])
       ys.append(value[1])

    for value in successPopulationValue:
        plt.plot(xs, ys, 'rx-')

    plt.scatter(bestChromosome[0],bestChromosome[1],
                c="red", marker = 'x', lw=1, label="best point")
    # plt.set(xlim=(minValue, maxValue), ylim=(minValue, maxValue))

    plt.legend()
    plt.show()
# In[10]: Main Program

# ******* setup *********
minValue = -100
maxValue = 100
target = "maximum" # can be "minimum", "maximum", or a value ex : 7.324
# ************************************
# ******* Algorithm parameters *********
maxGeneration = 1000 # timeout of calculation
populationSize = 10 # sample to work with (increasing number increases drasticaly calculation time)
chromosomeSize = 2 # number of parameters (Do not change for continuous)
mutationRate = 0.9 # for this kind of algorithm, higher mutation means better result

# ************************************

successChromosomeValue = []
successPopulationValue = []

minValue, maxValue, f = setupFunctionLimitation(minValue, maxValue)

population = InitialPopulation(populationSize, chromosomeSize, minValue, maxValue)
print(f"")
print(f" Initial population : {population}")

for i in range(maxGeneration):
    print(f"calculating... generation {i+1}")
    fitnessValues = Fitness(population, f)
# =============================================================================
#     print(f"")
#     print(f"fitness values : {fitnessValues}")
# =============================================================================

    targetValue = GoalCondition(target)

    if targetValue != None:
        fitnessProbability, fittestChromosome, fittestValue = FitnessProbability(fitnessValues, population, targetValue)
    # =========================================================================z====
    #     print(f"")
    #     print(f"probability table : {fitnessProbability}")
    # =============================================================================
        fittest = FittestSelection(population, fitnessProbability, populationSize)
    # =============================================================================
    #     print(f"")
    #     print(f"fittest : {fittest}")
    # =============================================================================

        randomParents = MateSelection(fittest, populationSize)
    # =============================================================================
    #     print(f"")
    #     print(f"Chromosomes Coupled : {randomParents}")
    # =============================================================================

        offpringsGeneration = MateHauptMethod(randomParents)
    # =============================================================================
    #     print(f"")
    #     print(f"Final offsprings generation : {offpringsGeneration}")
    # =============================================================================

        population = Mutation(fittest, offpringsGeneration, mutationRate, populationSize, chromosomeSize, minValue, maxValue)

        successPopulationValue.append(fittestValue)
        successChromosomeValue.append(fittestChromosome)
    else:
        print(f"")
        print(f" Can't continue calculation")
        break

print(f"")
print(f"calculating done")
print(f"")
print (f"Final population : {population}")
print(f"")


PlottingOfResult(maxGeneration, targetValue, successPopulationValue, successChromosomeValue, f, minValue, maxValue)