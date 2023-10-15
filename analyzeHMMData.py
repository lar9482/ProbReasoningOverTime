import pandas as pd
import matplotlib.pyplot as plt

def plotStateEstimation(evidenceTuple):
    filepath = constructAnalysisFilePath(evidenceTuple, 'stateEstimation')
    rawFileDF = pd.read_excel(filepath, engine='openpyxl')

    timeSteps = list(rawFileDF['T'])
    probs = list(rawFileDF['Prob'])

    plt.ylim(0, 1)
    plt.xlabel('time')
    plt.ylabel('Probability of the state being enoughSleep') 
    plt.title('Evidence: ' + evidenceTuple)
    plt.plot(timeSteps, probs)
    plt.savefig('./results/HMM/' + evidenceTuple + '.png')
    plt.clf()

def plotSmoothing(evidenceTuple):
    filepath = constructAnalysisFilePath(evidenceTuple, 'smoothingResults')
    rawFileDF = pd.read_excel(filepath, engine='openpyxl')

    timeSteps = list(rawFileDF['T'])
    countryDanceT2 = list(rawFileDF['CountryDance_T-2'])
    fixedLagT2 = list(rawFileDF['FixedLag_T-2'])

    countryDanceT3 = list(rawFileDF['CountryDance_T-3'])
    fixedLagT3 = list(rawFileDF['FixedLag_T-3'])

    countryDanceT4 = list(rawFileDF['CountryDance_T-4'])
    fixedLagT4 = list(rawFileDF['FixedLag_T-4'])

    countryDanceT5 = list(rawFileDF['CountryDance_T-5'])
    fixedLagT5 = list(rawFileDF['FixedLag_T-5'])
    
    plt.ylim(0, 1)
    plt.xlim(1, 25)
    plt.xlabel('T')
    plt.ylabel('Probability of the state being enoughSleep at T-2') 
    plt.title('Evidence: ' + evidenceTuple)
    plt.plot(timeSteps, countryDanceT2, label='countryDance')
    plt.plot(timeSteps, fixedLagT2, label='fixedLag')
    plt.legend(loc='upper right')
    
    plt.savefig('./results/HMM/' + evidenceTuple + 'T2.png')
    plt.clf()

    plt.ylim(0, 1)
    plt.xlim(1, 25)
    plt.xlabel('T')
    plt.ylabel('Probability of the state being enoughSleep at T-3') 
    plt.title('Evidence: ' + evidenceTuple)
    plt.plot(timeSteps, countryDanceT3, label='countryDance')
    plt.plot(timeSteps, fixedLagT3, label='fixedLag')
    plt.legend(loc='upper right')
    
    plt.savefig('./results/HMM/' + evidenceTuple + 'T3.png')
    plt.clf()

    plt.ylim(0, 1)
    plt.xlim(1, 25)
    plt.xlabel('T')
    plt.ylabel('Probability of the state being enoughSleep at T-4') 
    plt.title('Evidence: ' + evidenceTuple)
    plt.plot(timeSteps, countryDanceT4, label='countryDance')
    plt.plot(timeSteps, fixedLagT4, label='fixedLag')
    plt.legend(loc='upper right')
    
    plt.savefig('./results/HMM/' + evidenceTuple + 'T4.png')
    plt.clf()

    plt.ylim(0, 1)
    plt.xlim(1, 25)
    plt.xlabel('T')
    plt.ylabel('Probability of the state being enoughSleep at T-5') 
    plt.title('Evidence: ' + evidenceTuple)
    plt.plot(timeSteps, countryDanceT5, label='countryDance')
    plt.plot(timeSteps, fixedLagT5, label='fixedLag')
    plt.legend(loc='upper right')
    
    plt.savefig('./results/HMM/' + evidenceTuple + 'T5.png')
    plt.clf()

def constructAnalysisFilePath(evidenceTupleString, resultTypeString):
    return './results/HMM/{0}_{1}.xlsx'.format(evidenceTupleString, resultTypeString)
    
