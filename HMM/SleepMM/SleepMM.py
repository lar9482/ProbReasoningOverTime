from HMM.utils.HMM import HMM
from HMM.SleepMM.SleepMMVariables import Sleep_S, SleepInClass_E, RedEyes_E

import numpy as np

class SleepMM(HMM):

    def __init__(
        self,
        transTable,
        sensorTable,
        priorTable, 
        stateVariable,
        stateCardinality,
        transMatrix,
        priorMatrix
    ):      
        super().__init__(transTable, sensorTable, priorTable, stateVariable, stateCardinality)
        self.transMatrix = transMatrix
        self.priorMatrix = priorMatrix
    
    def lookUpSensor_GivenEvidenceAndStateValues(
        self, redEyesValue_Ev, sleepInClassValue_Ev, enoughSleepValue_St
    ):
        return self.sensorTable[(
            redEyesValue_Ev, sleepInClassValue_Ev, enoughSleepValue_St
        )]

    def lookUpTrans_GivenCurrAndLastTransitionValue(self, currStateValue, prevStateValue):
        return self.transTable[(prevStateValue, currStateValue)]
    
def getSleepMM():

    priorMatrix = np.zeros((len(Sleep_S.__members__), 1))
    priorMatrix[Sleep_S.enoughSleep.value] = 0.7
    priorMatrix[Sleep_S.notEnoughSleep.value] = 0.3

    priorTable = {
        Sleep_S.enoughSleep.value: 0.7,
        Sleep_S.notEnoughSleep.value: 0.3
    }

    # i: previous state value
    # j: current state value
    transitionMatrix = np.zeros((len(Sleep_S.__members__), len(Sleep_S.__members__)))
    transitionMatrix[Sleep_S.enoughSleep.value][Sleep_S.enoughSleep.value] = 0.8
    transitionMatrix[Sleep_S.enoughSleep.value][Sleep_S.notEnoughSleep.value] = 0.2
    transitionMatrix[Sleep_S.notEnoughSleep.value][Sleep_S.enoughSleep.value] = 0.3
    transitionMatrix[Sleep_S.notEnoughSleep.value][Sleep_S.notEnoughSleep.value] = 0.7

    # (previousStateValue, currentStateValue)
    transitionTable = {
        (Sleep_S.enoughSleep.value, Sleep_S.enoughSleep.value): 0.8,
        (Sleep_S.enoughSleep.value, Sleep_S.notEnoughSleep.value): 0.2,
        (Sleep_S.notEnoughSleep.value, Sleep_S.enoughSleep.value): 0.3,
        (Sleep_S.notEnoughSleep.value, Sleep_S.notEnoughSleep.value): 0.7
    }

    sensorTable = {
        (RedEyes_E.redEyes.value, SleepInClass_E.sleepInClass.value, Sleep_S.enoughSleep.value): 0.02,
        (RedEyes_E.redEyes.value, SleepInClass_E.sleepInClass.value, Sleep_S.notEnoughSleep.value): 0.21,
        (RedEyes_E.redEyes.value, SleepInClass_E.notSleepInClass.value, Sleep_S.enoughSleep.value): 0.18,
        (RedEyes_E.redEyes.value, SleepInClass_E.notSleepInClass.value, Sleep_S.notEnoughSleep.value): 0.49,
        (RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value, Sleep_S.enoughSleep.value): 0.08,
        (RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value, Sleep_S.notEnoughSleep.value): 0.09,
        (RedEyes_E.notRedEyes.value, SleepInClass_E.notSleepInClass.value, Sleep_S.enoughSleep.value): 0.72,
        (RedEyes_E.notRedEyes.value, SleepInClass_E.notSleepInClass.value, Sleep_S.notEnoughSleep.value): 0.22
    }

    return SleepMM(
        transitionTable, sensorTable, priorTable, Sleep_S, len(Sleep_S.__members__),
        transitionMatrix, priorMatrix
    )