import random
import copy

from CS5313_Localization_Env.localization_env import Headings
class RobotDBN:
    """    
        Variables that model the DBN:

        -- locationPriorTable:
        The prior probabilities for each free cell, given as a dictionary of 
        {(x1, y1) : prob, ..., (xn, yn) : prob} where each (xi, yi) is a free cell

        -- headingPriorTable:
        The prior probabilities for each heading, given as a dictionary of 
        {Heading.S : prob, ..., Heading.W : prob}

        -- locationTransTable:
        The table of transition probabilities for each cell. Format is [x][y][heading][direction] 
        which will return the probabilities of moving the direction, given the robot's current x, y, and heading.

        -- headingTransTable:
        The table of transition probabilities for the headings given each cell. Format is [x][y][heading][heading] 
        which will return the probabilities of each heading for the next time step given the robot's current x, y, and heading.

        -- sensorTable:
        The observation probabilities at any given location
        sensorTable[x][y][observation_tuple] = probability of that observation
    """
    def __init__(self, 
        env, N
    ):
        self.locationPriorTable = env.location_priors
        self.headingPriorTable = env.heading_priors
        self.locationTransTable = env.location_transitions
        self.headingTransTable = env.headings_transitions
        self.sensorTable = env.observation_tables

        self.N = N
        self.S = self.__generateInitialSamples()
    
    def __generateInitialSamples(self):
        
        priorSamplesAndProb = self.__constructPriorCombinedTable()
        initialSamples = []
        for _ in range(0, self.N):
            chance = random.uniform(0, 1)
            probSum = 0
            
            for prob in list(priorSamplesAndProb.keys()):
                if (probSum <= chance and chance < (probSum + prob)):
                    initialSamples.append(random.choice(priorSamplesAndProb[prob]))
                    break
                probSum += prob

        return initialSamples

    def __constructPriorCombinedTable(self):
        priorSamplesAndProb = {}
        for location in list(self.locationPriorTable.keys()):
            locationX = location[0]
            locationY = location[1]

            for heading in list(self.headingPriorTable.keys()):
                combinedProb = self.locationPriorTable[location] * self.headingPriorTable[heading]
                combinedSample = (locationX, locationY, heading)

                if (priorSamplesAndProb.get(combinedProb) == None):
                    priorSamplesAndProb[combinedProb] = [combinedSample]
                else:
                    priorSamplesAndProb[combinedProb].append(combinedSample)

        self.__normalizePriorCombinedTable(priorSamplesAndProb)
        return priorSamplesAndProb
    
    def __normalizePriorCombinedTable(self, priorSamplesAndProb):
        sumOfCombinedPriorProbs = sum(list(priorSamplesAndProb.keys()))
        for prob in list(priorSamplesAndProb.keys()):
            if (sumOfCombinedPriorProbs != 1.000):
                priorSamplesAndProb[prob / sumOfCombinedPriorProbs] = priorSamplesAndProb[prob]
                priorSamplesAndProb.pop(prob)


    ######################################################
    # Methods that pertain to particle filtering itself. #
    ######################################################
    def __buildTransLocationProbTable(self, x, y, heading):
        directionProbMap = self.locationTransTable[x][y][heading]
        transLocationProbMap = {}
        for direction in list(directionProbMap.keys()):
            directionX = direction.value[0]
            directionY = direction.value[1]

            transLocationProbMap[(x + directionX, y + directionY)] = directionProbMap[direction]

        return transLocationProbMap

    def __constructNextSampleProbTable(self, currSample):
        currX = currSample[0]
        currY = currSample[1]
        currHeading = currSample[2]

        transLocationProbTable = self.__buildTransLocationProbTable(currX, currY, currHeading)
        transHeadingProbTable = self.headingTransTable[currX][currY][currHeading]
        
        nextSampleProbTable = {}
        for location in list(transLocationProbTable.keys()):
            locationX = location[0]
            locationY = location[1]

            for heading in list(transHeadingProbTable.keys()):
                generatedSample = (locationX, locationY, heading)
                generatedSampleProb = transLocationProbTable[location] * transHeadingProbTable[heading]
                nextSampleProbTable[generatedSample] = generatedSampleProb

        return nextSampleProbTable
    
    def __sampleFromTransitionModel(self, currSample):
        nextSampleProbTable = self.__constructNextSampleProbTable(currSample)
        probSum = 0
        chance = random.uniform(0, 1)

        for sample in list(nextSampleProbTable.keys()):
            if (probSum <= chance and chance < (probSum + nextSampleProbTable[sample])):
                return sample
            probSum += nextSampleProbTable[sample]

    def __normalizeWeights(self, W):
        sumWeights = sum(W)

        for i in range(0, len(W)):
            W[i] = W[i] / sumWeights

    def __resampleWithWeights(self, W):
        self.__normalizeWeights(W)
        newSamples = []

        for i in range(0, self.N):
            chance = random.uniform(0, 1)
            probSum = 0

            for i in range(0, self.N):
                sampleWeight = W[i]
                if (probSum <= chance and chance < (sampleWeight+probSum)):
                    newSamples.append(self.S[i])
                    break
                probSum += sampleWeight
        
        return newSamples
    
    def runParticleFilter(self, evidence):
        W = []
        for i in range(0, self.N):
            self.S[i] = self.__sampleFromTransitionModel(self.S[i])
            S_ix = self.S[i][0]
            S_iy = self.S[i][1]

            W.append(
                self.sensorTable[S_ix][S_iy][tuple(evidence)]
            )

        self.S = self.__resampleWithWeights(W)
        return self.S

    
    def calcLocationProbsFromSamples(self, samples, dimensionX, dimensionY):

        #Given the samples, construct an initial table {location: prob} by counts
        locProbTable = {}
        for sample in samples:
            X = sample[0]
            Y = sample[1]
            if (locProbTable.get((X, Y)) == None):
                locProbTable[(X, Y)] = 1
            else:
                locProbTable[(X, Y)] += 1

        #Then, put this table into a 2d list defined by dimensionX, dimension Y
        twoDLocProbTable = [[0] * (dimensionY) for _ in range(dimensionX)]
        for XY in list(locProbTable.keys()):
            locProbTable[XY] = locProbTable[XY] / len(samples)

        # Normalizing the probability entries in locProbTable
        sumOfRawProbs = sum(locProbTable.values())
        for XY in list(locProbTable.keys()):
            rawProbOfXY = locProbTable[XY]
            normalizedProbOfXY = rawProbOfXY / sumOfRawProbs
            locProbTable[XY] = normalizedProbOfXY

            X = XY[0]
            Y = XY[1]
            twoDLocProbTable[X][Y] = locProbTable[XY]

        return twoDLocProbTable
    
    def calcHeadingProbsFromSamples(self, samples):
        #Given the samples, construct an initial table {heading: prob} by counts
        headingProbTable = {}
        for sample in samples:
            sampleHeading = sample[2]
            if (headingProbTable.get(sampleHeading) == None):
                headingProbTable[sampleHeading] = 1
            else:
                headingProbTable[sampleHeading] += 1
        
        for heading in list(headingProbTable.keys()):
            headingProbTable[heading] = headingProbTable[heading] / len(samples)
        
        # Normalizing the probability entries in headingProbTable
        for heading in list(headingProbTable.keys()):
            rawHeadingProb = headingProbTable[heading]
            normalizedProb = rawHeadingProb / len(Headings)
            headingProbTable[heading] = normalizedProb

        #Fill out the rest of the headings that aren't in the samples.
        for unusedHeading in Headings:
            if (headingProbTable.get(unusedHeading) == None):
                headingProbTable[unusedHeading] = 0

        return headingProbTable
    
    def getMostLikelySamples(self, samples):
        sampleProbTable = {}
        probSampleTable = {}

        # Counting occurences
        for sample in samples:
            if (sampleProbTable.get(sample) == None):
                sampleProbTable[sample] = 1
            else:
                sampleProbTable[sample] += 1

        # Grouping the count occurences into raw probabilities
        for sample in samples:
            prob = sampleProbTable[sample] / len(samples)

            if (probSampleTable.get(prob) == None):
                probSampleTable[prob] = [sample]
            
            elif(not sample in probSampleTable[prob]):
                probSampleTable[prob].append(sample)
        
        # Normalizing the raw probabilities
        sumProb = sum(list(probSampleTable.keys()))
        for rawProb in list(probSampleTable.keys()):
            normalizedProb = rawProb / sumProb
            probSampleTable[normalizedProb] = probSampleTable[rawProb]
            if (rawProb != normalizedProb):
                probSampleTable.pop(rawProb)
        
        maxProb = max(list(probSampleTable.keys()))
        return (maxProb, probSampleTable[maxProb])