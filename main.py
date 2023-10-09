from testHMM import testHMM
from testHMM import HMMEvidence

from CS5313_Localization_Env import localization_env as le
from DBN.RobotDBN import RobotDBN

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
    # testDBN()
    testHMM(HMMEvidence.redEyes_sleepClass)
    testHMM(HMMEvidence.redEyes_notSleepClass)
    testHMM(HMMEvidence.notRedEyes_sleepClass)
    testHMM(HMMEvidence.notRedEyes_notSleepClass)
    testHMM(HMMEvidence.random)

if __name__ == "__main__":
    main()