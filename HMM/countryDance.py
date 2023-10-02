from HMM.utils.HMMUtils import forward, backward, normalizeVector

import numpy as np
"""
    Performs:
        - Filtering for P(X_t+1 | E_1:t)
        - Smoothing for P(X_k | E_1:t) where 0 < k <= t

    using the forward-backward algorithm in matrix formulation

    @param HMM: 
        The hidden markov model, which defines a transition model and prior distribution

    @param evidenceValues: [....(E1...EN)]
        The list of evidence from time 1...t
        where the entries are tuples with the evidence values of all evidence variables.
    
    @param t:
        The current time step
    
    @returns [...np.array(S, 1)]
        A list of probability distributions from time 1...t, 
        where S is the number of values in the state variable.

        This represents the smoothed probability distributions from time 1...t
"""
def countryDance(HMM, evidenceValues, t):
    allForwardVectors = [HMM.priorMatrix]
    allSmoothVectors = []
    backwardVector = np.ones((HMM.stateCardinality, 1))

    # from 1..t
    for i in range(1, t+1):
        prevForwardVector = allForwardVectors[i-1]
        currEvidence = evidenceValues[i]
        allForwardVectors.append(
            forward(HMM, prevForwardVector, currEvidence)
        )

    # from t..1
    for i in range(t, 0, -1):
        currForwardVector = allForwardVectors[i]
        currSmoothVector = normalizeVector(
            np.multiply(currForwardVector, backwardVector)
        )
        currEvidence = evidenceValues[i]

        allSmoothVectors = [currSmoothVector, *allSmoothVectors]
        backwardVector = backward(HMM, backwardVector, currEvidence)
    
    return allSmoothVectors