from HMM.SleepMMVariables import Sleep_S
import numpy as np

def forward(HMM, prevForwardVector, currEvidence):
    observationMatrix = buildObservationMatrix(HMM, currEvidence)

    futureForwardVector = np.dot(
        np.dot(observationMatrix, HMM.transMatrix.transpose()),
        prevForwardVector
    )

    return normalizeVector(futureForwardVector)

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