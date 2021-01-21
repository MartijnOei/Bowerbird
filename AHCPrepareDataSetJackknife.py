'''
Martijn Simon Soen Liong Oei, April 12020 H.E.

Prepare jackknife data sets.
'''

import AHCFunctions, numpy, os

def AHCPrepareDataSetJackknife(directoryData, fileName, indexColumnStart, indexColumnEnd, numberOfDataSets, numberOfObservationsSubset, numberOfNumerals):
    '''
    '''

    # Load full data set.
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = AHCFunctions.loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)

    # Generate and save new data sets.
    for indexDataSet in range(numberOfDataSets):
        # Generate data set.
        indicesObservations     = numpy.random.choice(numpy.arange(numberOfObservations), size = numberOfObservationsSubset, replace = False)
        observationMatrixSubset = observationMatrix[indicesObservations]

        # Save data set.
        dataSetName             = "jackknife" + str(numberOfObservationsSubset) + "_" + str(indexDataSet).zfill(numberOfNumerals)
        directoryDataSet        = directoryData + dataSetName + "/"
        if (not os.path.exists(directoryDataSet)):
            os.makedirs(directoryDataSet)
        numpy.save(directoryDataSet + "observationMatrix.npy",   observationMatrixSubset)
        numpy.save(directoryDataSet + "indicesObservations.npy", indicesObservations)

        # Report progress.
        print("Saved data set " + dataSetName + " in " + directoryDataSet + "!")
