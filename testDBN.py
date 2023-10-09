from CS5313_Localization_Env import localization_env as le
from DBN.RobotDBN import RobotDBN
from openpyxl import Workbook

class testDBNParameters:
    def __init__(self, 
        dimension, 
        actionNoise, 
        observationNoise,
        actionBias,
        numParticles
    ):
        self.dimension = dimension
        self.actionNoise = actionNoise
        self.observationNoise = observationNoise
        self.actionBias = actionBias
        self.numParticles = numParticles

class timeStepResult:
    def __init__(self, t, mostLikelyState, prob, actualState):
        self.t = t
        self.mostLikelyState = mostLikelyState
        self.prob = prob
        self.actualState = actualState

def getTestDBNDatasetParameters():
    actionBiases = [-0.5, 0, 0.5]
    actionNoises = [0.1, 0.5, 0.9]
    observationNoises = [0.1, 0.5, 0.9]
    dimensions = [10, 20, 30]
    numParticles = [10, 100, 500, 1000]

    dataset = {}
    for numParticle in numParticles:
        for dimension in dimensions:
            for actionNoise in actionNoises:
                for observationNoise in observationNoises:
                    for actionBias in actionBiases:
                        testCase = testDBNParameters(
                            dimension,
                            actionNoise,
                            observationNoise,
                            actionBias,
                            numParticle
                        )

                        if (dataset.get(numParticle) == None):
                            dataset[numParticle] = [testCase]
                        else:
                            dataset[numParticle].append(testCase)
                        
    return dataset

def runDBNTest(testDBNParameter, totalTimeSteps):
    dimensionX = testDBNParameter.dimension
    dimensionY = testDBNParameter.dimension

    action_bias = testDBNParameter.actionBias
    observation_noise = testDBNParameter.observationNoise
    action_noise = testDBNParameter.actionNoise
    dimensions = (dimensionX, dimensionY)
    seed = 5000
    (x, y) = (750, 750)
    env = le.Environment(
        action_bias, 
        observation_noise, 
        action_noise, 
        dimensions, 
        seed=seed, 
        window_size=[x,y]
    )
    
    DBN = RobotDBN(env, testDBNParameter.numParticles)
    allTimeStepResults = []

    observation = env.observe()
    for t in range(1, totalTimeSteps+1):
        samples = DBN.runParticleFilter(observation)
        (prob, mostLikelySamples) = DBN.getMostLikelySamples(samples)

        allTimeStepResults.append(timeStepResult(
            t,
            mostLikelySamples,
            prob,
            (env.robot_location[0], env.robot_location[1], env.robot_heading)
        ))

        locProbs = DBN.calcLocationProbsFromSamples(samples, dimensionX, dimensionY)
        headingProbs = DBN.calcHeadingProbsFromSamples(samples)
        env.update(locProbs, headingProbs)
        observation = env.move()

    saveAllTimeStepResults(testDBNParameter, allTimeStepResults)

def saveAllTimeStepResults(testDBNParameter, allTimeStepResults):
    workbook = Workbook()
    sheet = workbook.active

    # Add headers
    sheet.append([
        'NumParticles: ' + str(testDBNParameter.numParticles),
        'Dimension: ' + str(testDBNParameter.dimension),
        'ActionNoise: '+ str(testDBNParameter.actionNoise),
        'ObservationNoise: ' + str(testDBNParameter.observationNoise),
        'ActionBias: ' + str(testDBNParameter.actionBias)
    ])
    sheet.append(['Most Likely State(s)', 'Prob', 'ActualState'])

    for timeStepResult in allTimeStepResults:
        sheet.append([
            str(timeStepResult.mostLikelyState),
            str(timeStepResult.prob),
            str(timeStepResult.actualState)
        ])
    
    fileName = 'particles_{0}-dim_{1}-actNoise_{2}-obsNoise_{3}-actBias{4}'.format(
        str(testDBNParameter.numParticles),
        str(testDBNParameter.dimension),
        str(testDBNParameter.actionNoise),
        str(testDBNParameter.observationNoise),
        str(testDBNParameter.actionBias)
    )

    workbook.save('./results/DBN/' + fileName + '.xlsx')