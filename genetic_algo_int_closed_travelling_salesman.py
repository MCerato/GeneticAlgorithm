# *****traitement des accents dans le texte****************
# -*- coding: utf-8 -*-

# =============================================================================
import random
import math
import matplotlib.pyplot as plt
# =============================================================================


# In[0]: Descritption

# =============================================================================
# class Name(dependencies):
#     """
#     Description
#     ----------
#     Solving the closed "traveling Salesman" problem
#     It has been considered a random starting city
#     It could be considered (and simpler) to fix the starting city
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

# In[1]: setup the environment test (if needed)
# setup a map with randomly placed cities
def setupRandomMap(chromosomeSize, dimX, dimY):
    coordCities = {}

    # coordCities.update({"city " + "0":[0, 0]})
    # for i in range(1, chromosomeSize):
    for i in range(chromosomeSize):
        coordCities.update({"city " + str(i):[random.randint(0, dimX), random.randint(0, dimY)]})# Dictionnary of random values with format : "city x":[X, Y]

    return coordCities

# setup a map with fixed placed cities
def setupFixedMap(chromosomeSize, dimX, dimY):
    coordCities = {}
    i = 0
    j = 0
    for cityNb in range(chromosomeSize):
        coordCities.update({"city " + str(cityNb):[i, j]})# Dictionnary of random values with format : "city x":[X, Y]
        i+=1
        if i == dimX-1:
            i = 0
            j += 1
        if j == dimY-1:
            j = 0
        print(f"city : {cityNb} | coordX : {j} | coordY {i} ")

    return coordCities

# In[2]: population intialization
def InitialPopulation(populationSize = 100, chromosomeSize = 10):
    population = []

    for chromosomeIndex in range(populationSize):
        population.append([sequence for sequence in range(chromosomeSize)]) # create a random pool of chromosomes (1st generation

    for chromosomeIndex in range(populationSize):
        random.shuffle(population[chromosomeIndex])

    return population


# In[3]: fitness function
def Fitness(population, coordCities):
    fitnessValues = []
    globalFitness = 0

# ================== reset of distance variable in between sequence ===========
    for i, chromosome in enumerate(population):
        distance = 0 # rest of the variable distance in between chromosomes
# =============================================================================

# ================== reset of distance variable in between sequence ===========
        for j, gene in enumerate(chromosome):

            try:
                distance += math.dist(coordCities["city " + str(chromosome[j+1])], coordCities["city " + str(gene)]) # distance calculation from city 0 to city n-1
            except IndexError:
                distance += math.dist(coordCities["city " + str(chromosome[0])], coordCities["city " + str(gene)]) # distance calculation from city n to city 0 (closed travelling salesman)
        fitnessValues.append(distance)# add the distance calculation in between cities in list
# =============================================================================

    return fitnessValues

# In[3]: condition of selection
def FitnessProbability(fitnessValues):
    fitnessProbability = []
    weightTable = []
# =============================================================================
    highestFitness = min(fitnessValues) # takes the shortest distance
    lowestFitness = max(fitnessValues) # takes the shortest distance
# =============================================================================

# /!\=======================================================================/!\
# give a probability according to a linear scale in between Highest fitness and
# lowest fitness. Values is between 0 and 1.
# 0 : The lowest fitness
# 1 : The highest fitness

    for i, chromosomeFitness in enumerate(fitnessValues):
        if highestFitness != lowestFitness:
            fitnessProbability.append((lowestFitness-chromosomeFitness) / (lowestFitness-highestFitness))
        else:
            fitnessProbability.append(1)
# =============================================================================
#         print(f"")
#         print(f"shortest path : {highestFitness} | longest path : {lowestFitness}")
#         print(f"Chromosome Number : {i} Chromosome : {population[i]} fitness value {fitnessValue[i]} fitness proba {fitnessProbability[i]} ")
# =============================================================================

# =============================================================================
# determine the shortest path for this run
    fittestIndex = fitnessValues.index(highestFitness) # find the table index of the shortest distance
    fittestValue = population[fittestIndex] # retrieves the best chromosome in population according to the index
# =============================================================================

# =============================================================================
#     for i in range(1, len(fitnessValues)+1):
#         for value in range(len(fitnessValues)+1, i, -1):
#             weightTable.append(i)
#
#     return weightTable
# =============================================================================

    return fittestValue, highestFitness, fitnessProbability

# In[4]: selection of the fittest
def FittestSelection(population, fitnessProbability, populationSize):
    fittest = []

    fittest = random.choices(population, weights = fitnessProbability, k = math.ceil(populationSize/2)) # selct the fittest randomly based on their fitness value (weighted).
    return fittest

# In[5]: mate selecion
def MateSelection(fittest, populationSize):
    randomParents = []

    for i in range(math.ceil(populationSize/4)):
        randomCouple = random.sample(fittest, 2)
        randomParents.append(randomCouple) # Random couple selection in the fittest group (no weight)
    return randomParents


# In[6]: mating of chromosomes
def MateCycleCrossover(randomParents):
    newGeneration = []

    for mates in randomParents:
        mom = mates[0]
        dad = mates[1]
        offprings1 = [None for i in range(len(mom))] # setting up values wich are necessary to set parents values to the right place
        offprings2 = [None for i in range(len(dad))] #

# ================ check if offspring1 complete ===============================
        for i in range(len(offprings1)):
            if offprings1[i] == None:
                offringsFilled = False
                break
            else:
                offringsFilled = True
# =============================================================================

        x = int(random.uniform(0, len(mom)))# choose a random index to start swapping genes (x)

        while offringsFilled is False: # if offspring1 is not entirely filled up yet
            while offprings1[x] is not None: # if index x already has a value
                x = int(random.uniform(0, chromosomeSize))# choose an other gene (index) to swap

            flipCoin = random.getrandbits(1) #  # generates randomly a "1" or a "0" (flip a coin)

            if bool(flipCoin) is True: # takes randomly dad's value or mom's value
                offprings1[x] = dad[x]
            else:
                offprings1[x] = mom[x]
# =============================================================================
#             print(f"index : {x} | mom[x] : {mom[x]} | offspring1 : {offprings1}")
# =============================================================================

            if bool(flipCoin) is False: # if mom has been chosen for first gene to be set in this cycle
                y = 0
                while y < len(dad):
                    if offprings1[x] == dad[y]:
                        x = y
                        y = 0
                        if offprings1[x] == None: # if the index in offspring is free
                            offprings1[x] = mom[x] # put mom's value again (if you put dad's, then you'll have twice the same value which is impossible)
                        else:
                            break #if the space in offspring is not free, then break the cycle and restart by choosing randomly mom or dad

                    else:
                        y += 1

            else:  # if dad has been chosen for first gene to be set in this cycle - same as mom
                y = 0
                while y < len(mom):
                # for value in mom:
                    if offprings1[x] == mom[y]:
                    # if offprings1[x] == value:
                        # x = mom.index(value)
                        x = y
                        y = 0
                        if offprings1[x] == None:
                            offprings1[x] = dad[x]
                        else:
                            break
                    else:
                        y += 1

# ================ check if offspring1 complete =================================
            for i in range(len(offprings1)):
                if offprings1[i] == None:
                    offringsFilled = False
                    break
                else:
                    offringsFilled = True
# =============================================================================

# ================ set offspring2 according to offspring1 ======================
        for x in range(len(mom)):
            if offprings1[x] == mom[x]:
                offprings2[x] = dad[x]
            else:
                offprings2[x] = mom[x]
# =============================================================================

        newGeneration.append(offprings1)
        newGeneration.append(offprings2)
    return newGeneration


# In[7]: random mutation
def Mutation(oldGeneration, offpringsGeneration, mutationRate, populationSize, chromosomeSize):

    # the objective is to swap randomly 2 gene in a chromosome
    mutatedChromosome =[]

    nbOfMutation = math.ceil((populationSize)*chromosomeSize*mutationRate)  # Nb of mutation in the next generation

    for i in range(nbOfMutation):
        indexOfChromosomeToMutate = int(random.uniform(0, len(offpringsGeneration)-1)) # getting the index of the chromosome to mutate in the population
        chromosomeToMutate = offpringsGeneration[indexOfChromosomeToMutate] # getting what chromosome to mutate according to the index


        indexOfGene1ToMutate = int(random.uniform(0, chromosomeSize-1)) # getting the index of the gene1 to mutate in the chromosome
        indexOfGene2ToMutate = int(random.uniform(0, chromosomeSize-1)) # getting the index of the gene2 to mutate in the chromosome

        while indexOfGene1ToMutate == indexOfGene2ToMutate:
            indexOfGene2ToMutate = int(random.uniform(0, chromosomeSize-1)) # getting the index of the gene to mutate in the chromosome

        gene1ToMutate = chromosomeToMutate[indexOfGene1ToMutate] # getting the gene1 to mutate in the chromosome
        gene2ToMutate = chromosomeToMutate[indexOfGene2ToMutate] # getting the gene2 to mutate in the chromosome

# ================ swapping genes ======================
        mutatedChromosome = chromosomeToMutate.copy()
        temp = gene1ToMutate
        mutatedChromosome[indexOfGene1ToMutate] = gene2ToMutate
        mutatedChromosome[indexOfGene2ToMutate] = temp
# =============================================================================

        offpringsGeneration[indexOfChromosomeToMutate] = mutatedChromosome
        oldGeneration.extend(offpringsGeneration)
        newPopulation = oldGeneration.copy()
    return newPopulation

# In[8]: success criteria
def SuccessCriteria(population, populationSize, chromosomeSize):
    #TBD
    pass


# In[9]: plotting results
def PlottingOfResult(maxGeneration, successPopulationValue, coordCities):
    citiesCoordX= []
    citiesCoordY= []

    plt.figure(figsize=(14, 6))
    XAxisValue = [val for val in range(0, maxGeneration)]
    for i in range(len(coordCities)):
        city = "city " + str(i)
        Coord = coordCities[city]
        citiesCoordX.append(Coord[0])
        citiesCoordY.append(Coord[1])

    plt.subplot(1, 2, 1) # 1 line, 2 col, subfigure 1
    plt.title("Value of shortest path ")
    plt.plot(XAxisValue, successPopulationValue, c="red", lw=1, label="plop")
    plt.ylabel("lenght value")
    plt.xlabel("Nb of generations")

    plt.subplot(1, 2, 2) # 1 line, 2 col, subfigure 2
    plt.title("map of cities")
    plt.scatter(citiesCoordX, citiesCoordY, c="blue", lw=1, label="cities")
    plt.ylabel("coordinates Y")
    plt.xlabel("coordinates X")

    plt.legend()
    plt.show()

# In[10]: Main Program

# ******* setup *********
dimX = 100
dimY = 100
# ************************************

# ******* Jeux de paramètres *********
maxGeneration = 10000 # timeout of calculation
populationSize = 4 # sample to work with (increasing number increases drasticaly calculation time)
chromosomeSize = 100 # number of cities
mutationRate = 0.001
# ************************************

successChromosomeValue = []
successPopulationValue = []

coordCities = setupRandomMap(chromosomeSize, dimX, dimY)
# coordCities = setupFixedMap(chromosomeSize, dimX, dimY)
print(f"")
print(f"cities coordinates : {coordCities}")

population = InitialPopulation(populationSize, chromosomeSize)
print(f"")
print(f" Initial population : {population}")
print(f"wait for calculation done...")
for i in range(maxGeneration):
    # print(f"calculating... generation {i+1}")

    fitnessValues = Fitness(population, coordCities)
# =============================================================================
#     print(f"")
#     print(f"best Value : {fittestValue}")
#     print(f"")
#     print(f"Probability table : {fitnessProbability}")
# =============================================================================

    fittestValue, highestFitness, fitnessProbability = FitnessProbability(fitnessValues)
# =============================================================================
#     print(f"")
#     print(f"fittest : {fittest}")
# =============================================================================

    fittest = FittestSelection(population, fitnessProbability, populationSize)
# =============================================================================
#     print(f"")
#     print(f"fittest : {fittest}")
# =============================================================================

    randomParents = MateSelection(fittest, populationSize)
# =============================================================================
#     print(f"")
#     print(f"cChromosomes Coupled : {randomParents}")
# =============================================================================
    offpringsGeneration = MateCycleCrossover(randomParents)
# =============================================================================
#     print(f"")
#     print(f"Final offsprings generation : {offpringsGeneration}")
# =============================================================================

    population = Mutation(fittest, offpringsGeneration, mutationRate, populationSize, chromosomeSize)
# =============================================================================
#     print(f"")
#     print (f"Final population : {population}")
# =============================================================================

    successPopulationValue.append(highestFitness)
    successChromosomeValue.append(fittestValue)

print(f"")
print(f"calculating done")
print (f"Final population : {population}")
print (f"shortest value found : {min(successPopulationValue)}")
print (f"shortest path found : {successChromosomeValue[successPopulationValue.index(min(successPopulationValue))]}")
PlottingOfResult(maxGeneration, successPopulationValue, coordCities)
