from testDBN import runDBNTest, getTestDBNDatasetParameters
from testHMM import testHMM, HMMEvidence

from analyzeDBNData import analyzeTimeStepResults
from multiprocessing import Process
from CS5313_Localization_Env.localization_env import le
from DBN.RobotDBN import RobotDBN
def runHMMTests():
    testHMM(HMMEvidence.redEyes_sleepClass)
    testHMM(HMMEvidence.redEyes_notSleepClass)
    testHMM(HMMEvidence.notRedEyes_sleepClass)
    testHMM(HMMEvidence.notRedEyes_notSleepClass)
    testHMM(HMMEvidence.random)

def runDBNTests():
    dbnDataset = getTestDBNDatasetParameters()
    totalTimeSteps = 1000

    for dimension in list(dbnDataset.keys()):
        testCases = dbnDataset[dimension]

        allProcesses = []
        for testCase in testCases:
            dbnProcess = Process(
                target=runDBNTest,
                args=(testCase, totalTimeSteps)
            )
            allProcesses.append(dbnProcess)
            
        for process in allProcesses:
            process.start()
        
        for process in allProcesses:
            process.join()

def analyzeDBNTests():
    dbnDataset = getTestDBNDatasetParameters()
    for dimension in list(dbnDataset.keys()):
        testCases = dbnDataset[dimension]
        for testCase in testCases:
            analyzeTimeStepResults(testCase)

def runToyDBN():
    dimensionX = 20
    dimensionY = 20

    action_bias = 0.5
    observation_noise = 0.95
    action_noise = 0.5
    dimensions = (dimensionX, dimensionY)
    seed = 768
    (x, y) = (750, 750)
    env = le.Environment(
        action_bias, 
        observation_noise, 
        action_noise, 
        dimensions, 
        seed=seed, 
        window_size=[x,y]
    )
    
    DBN = RobotDBN(env, 1000)

    observation = env.observe()
    for t in range(1, 9999+1):
        samples = DBN.runParticleFilter(observation)
        (prob, mostLikelySamples) = DBN.getMostLikelySamples(samples)

        locProbs = DBN.calcLocationProbsFromSamples(samples, dimensionX, dimensionY)
        headingProbs = DBN.calcHeadingProbsFromSamples(samples)
        env.update(locProbs, headingProbs)
        observation = env.move()

def main():
    dbnDataset = getTestDBNDatasetParameters()
    totalTimeSteps = 1000

    # runDBNTest(dbnDataset[10][0], totalTimeSteps)
    # analyzeTimeStepResults(dbnDataset[10][0])
    # runDBNTests()
    # analyzeDBNTests()
    runToyDBN()
if __name__ == "__main__":
    main()