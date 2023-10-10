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
    actionNoises = [0.01, 0.5, 0.95]
    observationNoises = [0.01, 0.5, 0.95]
    # dimensions = [10, 20, 30]
    # numParticles = [10, 100, 500, 1000]
    dimensions = [10]
    numParticles = [10]

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
    seed = 7680
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
            round(prob, 4),
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
        't',
        'prob', 
        'predicted x(s)', 
        'predicted y(s)', 
        'predicted heading(s)',
        'actual x',
        'actual y',
        'actual heading'
    ])

    for timeStepResult in allTimeStepResults:
        t = timeStepResult.t
        prob = timeStepResult.prob
        predictedXs = [state[0] for state in timeStepResult.mostLikelyState]
        predictedYs = [state[1] for state in timeStepResult.mostLikelyState]
        predictedHeadings = [state[2].name for state in timeStepResult.mostLikelyState]
        actualX = timeStepResult.actualState[0]
        actualY = timeStepResult.actualState[1]
        actualHeading = timeStepResult.actualState[2].name

        sheet.append([
            str(t),
            str(prob),
            str(predictedXs),
            str(predictedYs),
            str(predictedHeadings),
            str(actualX),
            str(actualY),
            str(actualHeading)
        ])
    
    saveFilePath = constructRawFilePath(testDBNParameter)

    workbook.save(saveFilePath)

def constructRawFilePath(testDBNParameter):
    fileName = 'particles_{0}-dim_{1}-actNoise_{2}-obsNoise_{3}-actBias{4}'.format(
        str(testDBNParameter.numParticles),
        str(testDBNParameter.dimension),
        str(testDBNParameter.actionNoise),
        str(testDBNParameter.observationNoise),
        str(testDBNParameter.actionBias)
    )

    filePath = './results/DBN/Dim{0}/{1}.xlsx'.format(
        str(testDBNParameter.dimension),
        fileName
    )

    return filePath