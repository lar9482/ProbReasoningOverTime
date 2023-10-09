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

def getTestDBNDatasetParameters():
    actionBiases = [0.1, 0.25, 0.5, 0.75, 0.9]
    actionNoises = [0.1, 0.25, 0.5, 0.75, 0.9]
    observationNoises = [0.1, 0.25, 0.5, 0.75, 0.9]
    dimensions = [5, 10, 15, 20, 25]

    numParticles = [10, 100, 500, 1000]

    dataset = {}
    for numParticle in numParticles:
        for dimension in dimensions:
            for actionNoise in actionNoises:
                for observationNoise in observationNoises:
                    for actionBias in actionBiases:
                        dataset[numParticle] = testDBNParameters(
                            dimension,
                            actionNoise,
                            observationNoise,
                            actionBias,
                            numParticle
                        )
    
    return dataset