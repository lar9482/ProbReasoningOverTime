from HMM.SleepMM import getSleepMM
from HMM.SleepMMVariables import SleepInClass_E, RedEyes_E

from countryDance import countryDance
def main():
    sleepMM = getSleepMM()
    evidenceTuple = (RedEyes_E.notRedEyes.value, SleepInClass_E.sleepInClass.value)
    time = 10

    for t in range(1, time+1):
        evidenceList = [evidenceTuple for _ in range(0, t+1)]
        countryDance(sleepMM, evidenceList, t)

if __name__ == "__main__":
    main()