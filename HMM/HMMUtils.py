from HMM.SleepMMVariables import Sleep_S
import numpy as np

"""
    Computes the new forward message vector from the equation
    f_(1:t+1) = αO_(t+1)T^(T)f_(1:t)
    @param HMM:
        The hidden markov model, which defines a transition model and prior distribution

    @param prevForwardVector: np.array((S x 1))
        The forward message vector from the previous time step
        where S is the number of values for the state variable.

    @currEvidence: (E1,..,EN)
        The values for the evidence at the current time step.

    @returns currForwardVector: np.array((S x 1))
        The current forward msg vector.
"""
def forward(HMM, prevForwardVector, currEvidence):
    observationMatrix = buildObservationMatrix(HMM, currEvidence)

    futureForwardVector = np.dot(
        np.dot(observationMatrix, HMM.transMatrix.transpose()),
        prevForwardVector
    )

    return normalizeVector(futureForwardVector)

"""
    Computes the backward message vector using the equation
    b_(k+1:t) = T O_(k+1) b_(k+2:t)
    
    @param HMM:
        The hidden markov model, which defines a transition model and prior distribution

    @param backwardVector: np.array((S x 1))
        The backward message vector from the next time step
        where S is the number of values for the state variable.

    @currEvidence: (E1,..,EN)
        The values for the evidence at the current time step.

    @returns currBackwardVector: np.array((S x 1))
        The current backward msg vector.
"""
def backward(HMM, backwardVector, currEvidence):
    observationMatrix = buildObservationMatrix(HMM, currEvidence)

    pastBackwardVector = np.dot(
        np.dot(HMM.transMatrix, observationMatrix),
        backwardVector
    )

    return pastBackwardVector

def buildObservationMatrix(HMM, currEvidence):
    redEyesValue = currEvidence[0]
    sleepInClassValue = currEvidence[0]
    observationMatrix = np.zeros((len(Sleep_S.__members__), len(Sleep_S.__members__)))
    i = 0
    for sleepStatePossibility in Sleep_S:
        sleepValue = sleepStatePossibility.value
        sensorProb = HMM.lookUpProb_GivenEvidenceAndStateValues(
            redEyesValue,  sleepInClassValue, sleepValue
        )
        observationMatrix[i][i] = sensorProb
        i += 1
    
    return observationMatrix

def normalizeVector(vector):
    vectorSum = np.sum(vector)

    for i in range(0, len(vector)):
        for j in range(0, len(vector[0])):
            vector[i][j] = vector[i][j] / vectorSum
    return vector