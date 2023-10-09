from testDBN import runDBNTest, getTestDBNDatasetParameters
from testHMM import testHMM, HMMEvidence
from multiprocessing import Process

def runHMMTests():
    testHMM(HMMEvidence.redEyes_sleepClass)
    testHMM(HMMEvidence.redEyes_notSleepClass)
    testHMM(HMMEvidence.notRedEyes_sleepClass)
    testHMM(HMMEvidence.notRedEyes_notSleepClass)
    testHMM(HMMEvidence.random)

def runDBNTests():
    dbnDataset = getTestDBNDatasetParameters()
    totalTimeSteps = 500

    for numParticles in list(dbnDataset.keys()):
        testCases = dbnDataset[numParticles]

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
            
def main():
    dbnDataset = getTestDBNDatasetParameters()
    totalTimeSteps = 1000

    # runDBNTest(dbnDataset[10][0], totalTimeSteps)
    runDBNTests()

if __name__ == "__main__":
    main()