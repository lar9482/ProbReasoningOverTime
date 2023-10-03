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
        sampledLocations = self.__getInitialLocationSamples()
        sampledHeadings = self.__getInitialHeadingSamples()
        samples = []
        for i in range(0, self.N):
            samples.append((
                sampledLocations[i][0], sampledLocations[i][1], sampledHeadings[i]
            ))
        
        return samples

    def __getInitialLocationSamples(self):
        sortedLocationToProbMap = sorted(self.locationPriorTable.items(), key=lambda x: x[1], reverse=True)
        locations = [locAndProb[0] for locAndProb in sortedLocationToProbMap]
        sampledLocations = []

        for _ in range(0, self.N):
            chance = random.uniform(0, 1)
            probSum = 0
            i = 0
            for locAndProb in sortedLocationToProbMap:
                probSum += locAndProb[1]
                if (chance < probSum):
                    sampledLocations.append(locations[i])
                    break
                i += 1
        
        return sampledLocations  
    
    def __getInitialHeadingSamples(self):
        sortedHeadingToProbMap = sorted(self.headingPriorTable.items(), key=lambda x: x[1], reverse=True)
        locations = [headAndProb[0] for headAndProb in sortedHeadingToProbMap]
        sampledHeadings = []

        for _ in range(0, self.N):
            chance = random.uniform(0, 1)
            probSum = 0
            i = 0
            for headingAndProb in sortedHeadingToProbMap:
                probSum += headingAndProb[1]
                if (chance < probSum):
                    sampledHeadings.append(locations[i])
                    break
                i += 1
        
        return sampledHeadings
    
    def runParticleFilter(evidence):
        pass