'''
Martijn Simon Soen Liong Oei, April 12020 H.E.

Prepare uniform data sets.
'''

import AHCFunctions, numpy, os

def AHCPrepareDataSetUniform(directoryData, fileName, indexColumnStart, indexColumnEnd, numberOfDataSets, numberOfNumerals):
    '''
    '''

    # Load data set.
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = AHCFunctions.loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)


    # Identify dimensions with discrete values.
    listDimensionIndicesDiscrete = []
    listDimensionValuesDiscrete  = []
    for indexDimension in range(numberOfDimensions):
        # Find the unique values along the dimension. Then filter out NaNs: 'numpy.nan == numpy.nan' returns False, as NaNs cannot be compared normally.
        uniqueValues = numpy.unique(observationMatrix[ : , indexDimension])
        uniqueValues = uniqueValues[numpy.logical_not(numpy.isnan(uniqueValues))]

        # If there are 10 or fewer different values along the dimension, the dimension is assumed to be discretised.
        if (uniqueValues.shape[0] <= 10):
            listDimensionIndicesDiscrete.append(indexDimension)
            listDimensionValuesDiscrete.append(uniqueValues)
            print(indexDimension, uniqueValues)

    print("listDimensionIndicesDiscrete:", listDimensionIndicesDiscrete)
    print("listDimensionValuesDiscrete:", listDimensionValuesDiscrete)


    # Generate and save new data sets.
    isNaNObservationMatrix       = numpy.isnan(observationMatrix)

    for indexDataSet in range(numberOfDataSets):
        # Create a copy of 'observationMatrix'.
        observationMatrixUniform = numpy.copy(observationMatrix)

        # Fill all non-NaN values with draws from the continuous uniform distribution over (0, 1).
        observationMatrixUniform[numpy.logical_not(isNaNObservationMatrix)] = numpy.random.uniform(low = 0, high = 1, size = numberOfObservations * numberOfDimensions - numpy.sum(isNaNObservationMatrix))

        # Overwrite the entries of the discretised dimensions.
        for indexDimension, valuesDimension in zip(listDimensionIndicesDiscrete, listDimensionValuesDiscrete):
            isNaNDimension        = numpy.isnan(observationMatrix[ : , indexDimension])
            numberOfNaNsDimension = numpy.sum(isNaNDimension)
            observationMatrixUniform[numpy.logical_not(isNaNDimension), indexDimension] = numpy.random.choice(valuesDimension, size = numberOfObservations - numberOfNaNsDimension)

        # Save data set.
        dataSetName             = "uniform" + "_" + str(indexDataSet).zfill(numberOfNumerals)
        directoryDataSet        = directoryData + dataSetName + "/"
        if (not os.path.exists(directoryDataSet)):
            os.makedirs(directoryDataSet)
        numpy.save(directoryDataSet + "observationMatrix.npy", observationMatrixUniform)

        # Report progress.
        print("Saved data set " + dataSetName + " in " + directoryDataSet + "!")
