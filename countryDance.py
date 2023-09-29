from HMM.SleepMMVariables import Sleep_S
import numpy as np
"""
    @param HMM: 
        The hidden markov model, which defines a transition model and prior distribution

    @ev:
        A vector of evidence from time 1...t
    
    @t:
        The current time step
"""
def countryDance(HMM, ev, t):
    allForwardVectors = [HMM.priorMatrix]
    allSmoothVectors = []
    backwardVector = np.ones((len(Sleep_S.__members__), 1))

    for i in range(1, t+1):
        prevForwardVector = allForwardVectors[i-1]
        currEvidence = ev[i]
        allForwardVectors.append(
            forward(HMM, prevForwardVector, currEvidence)
        )
    
    for i in range(t, 0, -1):
        currForwardVector = allForwardVectors[i]
        currSmoothVector = normalizeVector(
            np.multiply(currForwardVector, backwardVector)
        )
        currEvidence = ev[i]

        allSmoothVectors = [currSmoothVector, *allSmoothVectors]
        backwardVector = backward(HMM, backwardVector, currEvidence)
    
    return allSmoothVectors
        

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