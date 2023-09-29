from HMM.SleepMMVariables import Sleep_S

import numpy as np

class FixedLagSmoothing():
    def __init__(self, HMM, lagValue):
        self.HMM = HMM
        self.lagValue = lagValue
        
        self.t = 1
        self.forwardMsg = HMM.priorMatrix
        self.dBackwardTransform = np.identity(len(Sleep_S.__members__))
        evidenceQueue = []

    def runSmoothing(evidenceT):
        pass