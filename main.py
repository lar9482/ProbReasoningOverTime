from HMM.SleepMM.SleepMM import getSleepMM
from HMM.SleepMM.SleepMMVariables import SleepInClass_E, RedEyes_E

from HMM.countryDance import countryDance
from HMM.fixedLagSmoothing import FixedLagSmoothing
from HMM.viterbiAlgorithm import viterbiAlgorithm

from CS5313_Localization_Env import localization_env as le
from DBN.RobotDBN import RobotDBN

import random
import numpy as np

def testHMM():
    sleepMM = getSleepMM()
    allEvidenceTuple = [
        (RedEyes_E.redEyes.value, SleepInClass_E.sleepInClass.value),
        (RedEyes_E.redEyes.value, SleepInClass_E.notSleepInClass.value),
        (RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value),
        (RedEyes_E.notRedEyes.value, SleepInClass_E.notSleepInClass.value)
    ]

    time = 25
    completeEvidenceList = [(RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value) for _ in range(0, time+1)]
    # completeEvidenceList = [random.choice(allEvidenceTuple) for _ in range(0, time+1)]

    twoLagSmoothing = FixedLagSmoothing(sleepMM, 2)
    threeLagSmoothing = FixedLagSmoothing(sleepMM, 3)
    fourLagSmoothing = FixedLagSmoothing(sleepMM, 4)
    fiveLagSmoothing = FixedLagSmoothing(sleepMM, 5)

    for t in range(1, time+1):
        # Padding the evidenceList with a value at time 0
        # in order to begin indexing at time 1.
        evidenceFrom1ToT = ["noEvidenceAt0"] + completeEvidenceList[1:t+1]

        distroFrom1ToT = countryDance(sleepMM, evidenceFrom1ToT, t)
        distroAt_T2 = twoLagSmoothing.runSmoothing(completeEvidenceList[t])
        distroAt_T3 = threeLagSmoothing.runSmoothing(completeEvidenceList[t])
        distroAt_T4 = fourLagSmoothing.runSmoothing(completeEvidenceList[t])
        distroAt_T5 = fiveLagSmoothing.runSmoothing(completeEvidenceList[t])
        mostLikelyPath = viterbiAlgorithm(sleepMM, evidenceFrom1ToT)
        
        print(distroFrom1ToT)
        print(mostLikelyPath)
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

def testDBN():
    dimensionX = 15
    dimensionY = 15

    action_bias = 0.75
    observation_noise = 0.15
    action_noise = 0.15
    dimensions = (dimensionX, dimensionY)
    seed = 5
    (x, y) = (750, 750)
    env = le.Environment(
        action_bias, 
        observation_noise, 
        action_noise, 
        dimensions, 
        seed=seed, 
        window_size=[x,y]
    )
    
    DBN = RobotDBN(env, 100)
    observation = env.observe()
    for _ in range(0, 25000):
        samples = DBN.runParticleFilter(observation)
        locProbs = DBN.calcLocationProbsFromSamples(samples, dimensionX, dimensionY)
        headingProbs = DBN.calcHeadingProbsFromSamples(samples)

        env.update(locProbs, headingProbs)
        observation = env.move()

def main():
    testDBN()
    # testHMM()

if __name__ == "__main__":
    main()