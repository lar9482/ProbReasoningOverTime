from testDBN import constructRawFilePath

import pandas as pd
import math

def analyzeTimeStepResults(testDBNParameter):
    filePath = constructRawFilePath(testDBNParameter)
    fileDF = pd.read_excel(filePath, engine='openpyxl')
    print(fileDF)
    calculatedDistances = getDistBetweenPredictedAndActual(fileDF)
    (correctLocations, correctHeadings) = getCorrectStateCount(fileDF)
    print()

def getDistBetweenPredictedAndActual(fileDF):
    predictedX = getColumnData(fileDF, 'predicted x(s)')
    predictedY = getColumnData(fileDF, 'predicted y(s)')

    actualX = getColumnData(fileDF, 'actual x')
    actualY = getColumnData(fileDF, 'actual y')

    distances = []
    for i in range(0, len(fileDF)):
        numPredicted = len(predictedX[i])
        currActualX = actualX[i]
        currActualY = actualY[i]
        for j in range(0, numPredicted):
            currPredictedX = predictedX[i][j]
            currPredictedY = predictedY[i][j]

            currDistance = math.sqrt(
                (currActualX - currPredictedX) ** 2
                +
                (currActualY - currPredictedY) ** 2
            )

            distances.append(currDistance)
    
    return distances

def getCorrectStateCount(fileDF):
    predictedX = getColumnData(fileDF, 'predicted x(s)')
    predictedY = getColumnData(fileDF, 'predicted y(s)')
    predictedHeadings = getColumnData(fileDF, 'predicted heading(s)')

    actualX = getColumnData(fileDF, 'actual x')
    actualY = getColumnData(fileDF, 'actual y')
    actualHeading = getColumnData(fileDF, 'actual heading')

    correctLocations = 0
    correctHeadings = 0

    for i in range(0, len(fileDF)):
        numPredicted = len(predictedX[i])
        currActualX = actualX[i]
        currActualY = actualY[i]
        currActualHeading = actualHeading[i]
        for j in range(0, numPredicted):
            currPredictedX = predictedX[i][j]
            currPredictedY = predictedY[i][j]
            currPredictedHeading = predictedHeadings[i][j]

            if ((currActualX == currPredictedX) and (currActualY == currPredictedY)):
                correctLocations += 1
            
            if (currActualHeading == currPredictedHeading):
                correctHeadings += 1

    return (correctLocations, correctHeadings)

def getColumnData(fileDF, columnName):
    if (columnName == 'predicted heading(s)'):
        rawSubColumn = [row.strip('[]').split(',') for row in list(fileDF[columnName])]
        subColumnData = []
        for columnEntry in rawSubColumn:
            fixedColumnEntry = []
            for headingEntry in columnEntry:
                strippedSingleQuoteEntry = headingEntry.strip('\'\'')
                fixedColumnEntry.append(strippedSingleQuoteEntry)
            
            subColumnData.append(fixedColumnEntry)
            
        return subColumnData

    elif (columnName == 'predicted x(s)' or columnName == 'predicted y(s)'):
        rawSubColumn = [row.strip('[]').split(',') for row in list(fileDF[columnName])]
        subColumnData = []
        for columnEntry in rawSubColumn:
            fixedColumnEntry = []
            for stringNumEntry in columnEntry:
                intNumEntry = int(stringNumEntry)
                fixedColumnEntry.append(intNumEntry)
                
            subColumnData.append(fixedColumnEntry)
        
        return subColumnData
    else:
        return list(fileDF[columnName])