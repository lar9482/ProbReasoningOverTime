from enum import Enum

# The hidden state variable, gotten enough sleep or not
class Sleep_S(Enum):
    enoughSleep = 0
    notEnoughSleep = 1

# The evidence variable of having red eyes or not
class RedEyes_E(Enum): 
    redEyes = 0
    notRedEyes = 1

# The evidence variable of sleeping in class or not
class SleepInClass_E(Enum):
    sleepInClass = 0
    notSleepInClass = 1