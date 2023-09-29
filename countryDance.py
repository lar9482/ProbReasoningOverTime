from HMM.HMMUtils import forward, backward, buildObservationMatrix, normalizeVector
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