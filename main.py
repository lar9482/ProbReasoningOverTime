from testDBN import runDBNTest, getTestDBNDatasetParameters
from testHMM import testHMM, HMMEvidence

from analyzeDBNData import analyzeTimeStepResults, graphTopDistanceAndProbFromDBN
from analyzeHMMData import plotStateEstimation, plotSmoothing
from multiprocessing import Process

def runHMMTests():
    testHMM(HMMEvidence.redEyes_sleepClass)
    testHMM(HMMEvidence.redEyes_notSleepClass)
    testHMM(HMMEvidence.notRedEyes_sleepClass)
    testHMM(HMMEvidence.notRedEyes_notSleepClass)
    testHMM(HMMEvidence.random)

def runDBNTests():
    dbnDataset = getTestDBNDatasetParameters()
    totalTimeSteps = 100

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

def analyzeHMMData():
    evidenceTuples = ['notRedEyes_notSleepClass', 'random']
    for evidenceTuple in evidenceTuples:
        plotStateEstimation(evidenceTuple)
        plotSmoothing(evidenceTuple)

def main():
    dbnDataset = getTestDBNDatasetParameters()
    totalTimeSteps = 1000

    # runDBNTest(dbnDataset[10][0], totalTimeSteps)
    # analyzeTimeStepResults(dbnDataset[10][0])
    # runDBNTests()
    # analyzeHMMData()
    graphTopDistanceAndProbFromDBN(10)
    graphTopDistanceAndProbFromDBN(20)
    graphTopDistanceAndProbFromDBN(30)


if __name__ == "__main__":
    main()