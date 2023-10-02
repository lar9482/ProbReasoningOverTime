from HMM.utils.HMMUtils import buildObservationMatrix, forward, normalizeVector
from collections import deque
import numpy as np

"""
    Performs smoothing for P(X_t-d) given P(X_t | E_1:t)
"""
class FixedLagSmoothing():
    def __init__(self, HMM, lagValue):
        self.HMM = HMM
        self.lagValue = lagValue
        
        self.t = 1
        self.forwardMsg = HMM.priorMatrix
        self.dBackwardTransform = np.identity(HMM.stateCardinality)
        self.evidenceQueue = deque()

    def runSmoothing(self, evidenceT):
        self.evidenceQueue.append(evidenceT)
        
        Observation_t = buildObservationMatrix(self.HMM, self.evidenceQueue[len(self.evidenceQueue)-1])
        
        if (self.t > self.lagValue):
            self.forwardMsg = forward(self.HMM, self.forwardMsg, self.evidenceQueue[0])
            self.evidenceQueue.popleft()

            Observation_td = buildObservationMatrix(self.HMM, self.evidenceQueue[0])
            interBackwardTransform = np.dot(
                np.dot(self.dBackwardTransform, self.HMM.transMatrix),
                Observation_t
            )
            self.dBackwardTransform = np.dot(
                np.dot(np.linalg.inv(Observation_td), np.linalg.inv(self.HMM.transMatrix)),
                interBackwardTransform
            )
        else:
            self.dBackwardTransform = np.dot(
                np.dot(self.dBackwardTransform, self.HMM.transMatrix),
                Observation_t
            )
        
        self.t += 1
        if (self.t > (self.lagValue+1)):
            backwardMsg = np.dot(
               self.dBackwardTransform,
               np.ones((self.HMM.stateCardinality, 1)) 
            )
            forwardBackwardVector = np.abs(np.multiply(self.forwardMsg, backwardMsg))

            return normalizeVector(forwardBackwardVector)
        else:
            return None