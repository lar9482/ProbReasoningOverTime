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
    completeEvidenceList = [(RedEyes_E.redEyes.value, SleepInClass_E.notSleepInClass.value) for _ in range(0, time+1)]
    # completeEvidenceList = [random.choice(allEvidenceTuple) for _ in range(0, time+1)]

    twoLagSmoothing = FixedLagSmoothing(sleepMM, 2)
    threeLagSmoothing = FixedLagSmoothing(sleepMM, 3)
    fourLagSmoothing = FixedLagSmoothing(sleepMM, 4)
    fiveLagSmoothing = FixedLagSmoothing(sleepMM, 5)

    for t in range(1, time+1):
        evidenceFrom0ToT = completeEvidenceList[0:t+1]

        distroFrom1ToT = countryDance(sleepMM, evidenceFrom0ToT, t)
        distroAt_T2 = twoLagSmoothing.runSmoothing(completeEvidenceList[t])
        distroAt_T3 = threeLagSmoothing.runSmoothing(completeEvidenceList[t])
        distroAt_T4 = fourLagSmoothing.runSmoothing(completeEvidenceList[t])
        distroAt_T5 = fiveLagSmoothing.runSmoothing(completeEvidenceList[t])

        if (isinstance(distroAt_T2, np.ndarray)):
            print('From Country Dance(2)')
            print(distroFrom1ToT[t-twoLagSmoothing.lagValue-1])
            print('From fixed Lag(2)')
            print(distroAt_T2)
        print()
        if (isinstance(distroAt_T3, np.ndarray)):
            print('From Country Dance(3)')
            print(distroFrom1ToT[t-threeLagSmoothing.lagValue-1])
            print('From fixed Lag(3)')
            print(distroAt_T3)
        print()
        if(isinstance(distroAt_T4, np.ndarray)):
            print('From Country Dance(4)')
            print(distroFrom1ToT[t-fourLagSmoothing.lagValue-1])
            print('From fixed Lag(4)')
            print(distroAt_T4)
        print()
        if(isinstance(distroAt_T5, np.ndarray)):
            print('From Country Dance(5)')
            print(distroFrom1ToT[t-fiveLagSmoothing.lagValue-1])
            print('From fixed Lag(5)')
            print(distroAt_T5)
        print()

if __name__ == "__main__":
    main()