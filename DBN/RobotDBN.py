import random

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
        
        samplesAndProb = self.__constructPriorCombinedTable()
        initialSamples = []
        for _ in range(0, self.N):
            chance = random.uniform(0, 1)
            probSum = 0
            for prob in list(samplesAndProb.keys()):
                if (probSum <= chance and chance < (probSum + prob)):
                    initialSamples.append(random.choice(samplesAndProb[prob]))
                    break
                probSum += prob

        return initialSamples

    def __constructPriorCombinedTable(self):
        samplesAndProb = {}
        for location in list(self.locationPriorTable.keys()):
            for heading in list(self.headingPriorTable.keys()):
                combinedProb = self.locationPriorTable[location] * self.headingPriorTable[heading]
                combinedSample = (location[0], location[1], heading)

                if (samplesAndProb.get(combinedProb) == None):
                    samplesAndProb[combinedProb] = [combinedSample]
                else:
                    samplesAndProb[combinedProb].append(combinedSample)

        self.__normalizePriorCombinedTable(samplesAndProb)
        return samplesAndProb
    
    def __normalizePriorCombinedTable(self, samplesAndProb):
        sumOfCombinedPriorProbs = sum(list(samplesAndProb.keys()))
        for prob in list(samplesAndProb.keys()):
            if (sumOfCombinedPriorProbs != 1.000):
                samplesAndProb[prob / sumOfCombinedPriorProbs] = samplesAndProb[prob]
                samplesAndProb.pop(prob)


    ######################################################
    # Methods that pertain to particle filtering itself. #
    ######################################################
    def __generateTransLocationAndProbTable(self, x, y, heading):
        directionProbMap = self.locationTransTable[x][y][heading]
        locationProbMap = {}
        for direction in list(directionProbMap.keys()):
            locationProbMap[
                (x + direction.value[0], y + direction.value[1])
            ] = directionProbMap[direction]

        return locationProbMap

    def __constructFutureSampleProbTable(self, currSample):
        locationProbTable = self.__generateTransLocationAndProbTable(currSample[0], currSample[1], currSample[2])
        headingProbTable = self.headingTransTable[currSample[0]][currSample[1]][currSample[2]]
        
        sampleProbTable = {}
        for location in list(locationProbTable.keys()):
            for heading in list(headingProbTable.keys()):
                generatedSample = (location[0], location[1], heading)
                generatedSampleProb = locationProbTable[location] * headingProbTable[heading]
                sampleProbTable[generatedSample] = generatedSampleProb

        return sampleProbTable
    
    def __sampleFromTransitionModel(self, currSample):
        sampleProbTable = self.__constructFutureSampleProbTable(currSample)
        probSum = 0

        random.seed(random.randint(0, 99999999))
        chance = random.uniform(0, 1)

        for sample in list(sampleProbTable.keys()):
            if (probSum <= chance and chance < (probSum + sampleProbTable[sample])):
                return sample
            probSum += sampleProbTable[sample]

    def __normalizeWeights(self, W):
        sumWeights = sum(W)

        for i in range(0, len(W)):
            W[i] = W[i] / sumWeights

    def resampleWithWeights(self, W):
        self.__normalizeWeights(W)
        for i in range(0, self.N):
            pass

    def runParticleFilter(self, evidence):
        W = []
        for i in range(0, self.N):
            self.S[i] = self.__sampleFromTransitionModel(self.S[i])
            W.append(self.sensorTable[self.S[i][0]][self.S[i][1]][tuple(evidence)])