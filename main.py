from HMM.SleepMM import getSleepMM
from HMM.SleepMMVariables import SleepInClass_E, RedEyes_E

from countryDance import countryDance
from fixedLagSmoothing import FixedLagSmoothing
import random
import numpy as np

def main():
    testMatrix = np.array([[
        1,2
    ], [
        3,4
    ]])
    testInverse = np.linalg.inv(testMatrix)
    print(np.dot(testMatrix, testInverse))

    sleepMM = getSleepMM()
    allEvidenceTuple = [
        (RedEyes_E.redEyes.value, SleepInClass_E.sleepInClass.value),
        (RedEyes_E.redEyes.value, SleepInClass_E.notSleepInClass.value),
        (RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value),
        (RedEyes_E.notRedEyes.value, SleepInClass_E.notSleepInClass.value)
    ]

    time = 25
    completeEvidenceList = [random.choice(allEvidenceTuple) for _ in range(0, time+1)]

    twoLagSmoothing = FixedLagSmoothing(sleepMM, 2)
    for t in range(1, time+1):
        evidenceFrom0ToT = completeEvidenceList[0:t+1]

        test1 = countryDance(sleepMM, evidenceFrom0ToT, t)
        test2 = twoLagSmoothing.runSmoothing(completeEvidenceList[t])
        if (isinstance(test2, np.ndarray)):
            print(test1[t-twoLagSmoothing.lagValue+1])
            print(test2)
        print()

if __name__ == "__main__":
    main()