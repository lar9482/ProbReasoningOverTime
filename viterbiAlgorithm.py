import sys

def viterbiAlgorithm(HMM, evidenceFrom1ToN):
    N = len(evidenceFrom1ToN)-1

    # Padding both M and A with a value at time 0
    # in order to begin indexing at 1.

    # The maximum probability of a path starting at any x0 and the evidence seen so far to a given xt at time t
    M = ['None']
    # The last transition along the best path to x_t 
    A = ['None']

    for t in range(1, N+1):
        redEyeValueAtTimeT = evidenceFrom1ToN[t][0]
        sleepInClassValueAtTimeT = evidenceFrom1ToN[t][1]

        M.append({})
        A.append({})

        for stateT in HMM.stateVariable:
            stateTValue = stateT.value
            if (t == 1):
                M[t][stateTValue] = (
                    HMM.priorTable[stateTValue]
                  * HMM.lookUpSensor_GivenEvidenceAndStateValues(redEyeValueAtTimeT, sleepInClassValueAtTimeT, stateTValue)
                )
                 
            else:
                A[t][stateTValue] = argMax_LastTransitionAlongBestPath(HMM, M, stateTValue, t)
                M[t][stateTValue] = (
                    HMM.lookUpSensor_GivenEvidenceAndStateValues(redEyeValueAtTimeT, sleepInClassValueAtTimeT, stateTValue)
                  * HMM.lookUpTrans_GivenCurrAndLastTransitionValue(stateTValue, A[t][stateTValue])
                  * M[t-1][A[t][stateTValue]]
                )
        
    bestPath = [getArgMaxBestLastState(HMM, M, N)]
    for t in range(N, 1, -1):
        bestPath = [A[t][bestPath[0]]] + bestPath

    return bestPath

def argMax_LastTransitionAlongBestPath(HMM, M, currStateTValue, t):
    argMaxPrevStateT_1 = -1
    maxTransitionProb = sys.float_info.min

    for prevStateT_1 in HMM.stateVariable:
        prevStateT_1Value = prevStateT_1.value

        transitionProb = (
            HMM.lookUpTrans_GivenCurrAndLastTransitionValue(currStateTValue, prevStateT_1Value)
          * M[t-1][prevStateT_1Value]    
        )

        if (transitionProb > maxTransitionProb):
            maxTransitionProb = transitionProb
            argMaxPrevStateT_1 = prevStateT_1Value
    
    return argMaxPrevStateT_1
        
def getArgMaxBestLastState(HMM, M, N):
    argMaxBestLastState = -1
    bestLastStateProb = -1

    for state in HMM.stateVariable:
        stateValue = state.value
        lastStateProb = M[N][stateValue]

        if (lastStateProb > bestLastStateProb):
            bestLastStateProb = lastStateProb
            argMaxBestLastState = stateValue
    
    return argMaxBestLastState