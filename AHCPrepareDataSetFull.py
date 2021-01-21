'''
Martijn Simon Soen Liong Oei, April 12020 H.E.

Prepare full data set.
'''

import AHCFunctions, numpy, os

def AHCPrepareDataSetFull(directoryData, fileName, indexColumnStart, indexColumnEnd, dataSetName = "full"):
    '''
    '''
    # Load data set.
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = AHCFunctions.loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)

    # Save data set.
    directoryDataSet = directoryData + dataSetName + "/"
    if (not os.path.exists(directoryDataSet)):
        os.makedirs(directoryDataSet)
    numpy.save(directoryDataSet + "observationMatrix.npy", observationMatrix)
