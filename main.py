from HMM.SleepMM import getSleepMM
from HMM.SleepMMVariables import SleepInClass_E, RedEyes_E

from countryDance import countryDance
from fixedLagSmoothing import FixedLagSmoothing
import random
import numpy as np

def main():

    sleepMM = getSleepMM()
    allEvidenceTuple = [
        (RedEyes_E.redEyes.value, SleepInClass_E.sleepInClass.value),
        (RedEyes_E.redEyes.value, SleepInClass_E.notSleepInClass.value),
        (RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value),
        (RedEyes_E.notRedEyes.value, SleepInClass_E.notSleepInClass.value)
    ]

    time = 25
    completeEvidenceList = [(RedEyes_E.notRedEyes.value, SleepInClass_E.notSleepInClass.value) for _ in range(0, time+1)]

    twoLagSmoothing = FixedLagSmoothing(sleepMM, 2)
    threeLagSmoothing = FixedLagSmoothing(sleepMM, 3)
    fourLagSmoothing = FixedLagSmoothing(sleepMM, 4)
    
    for t in range(1, time+1):
        evidenceFrom0ToT = completeEvidenceList[0:t+1]

        test1 = countryDance(sleepMM, evidenceFrom0ToT, t)
        test2 = twoLagSmoothing.runSmoothing(completeEvidenceList[t])
        test3 = threeLagSmoothing.runSmoothing(completeEvidenceList[t])
        test4 = fourLagSmoothing.runSmoothing(completeEvidenceList[t])
        if (isinstance(test2, np.ndarray)):
            print('From Country Dance(2)')
            print('From fixed Lag(2)')
            print(test1[t-twoLagSmoothing.lagValue-1])
            print(test2)
        print()
        if (isinstance(test3, np.ndarray)):
            print('From Country Dance(3)')
            print(test1[t-threeLagSmoothing.lagValue-1])
            print('From fixed Lag(3)')
            print(test3)
        print()
        if(isinstance(test4, np.ndarray)):
            print('From Country Dance(4)')
            print(test1[t-fourLagSmoothing.lagValue-1])
            print('From fixed Lag(4)')
            print(test4)
        print()

if __name__ == "__main__":
    main()