'''
Martijn Oei, February 12020 H.E.
'''

import numpy, pandas


def loadObservationMatrix(filePath, dimensionIndexStart, dimensionIndexEnd, sep = ";"):
    '''
    Load the observation matrix from a .csv-file at 'filePath',
    including only columns indexed by 'dimensionIndexStart' up to (but not including) 'dimensionIndexEnd'.

    observationMatrix: numpy.ndarray of floats, shape  '(numberOfObservations, numberOfDimensions)'
    dimensionsUsed:    list of strings,         length 'numberOfDimensions'
    '''

    dataFrame            = pandas.read_csv(filePath, sep = sep)
    dataFrame.replace(["1/3", "2/3"], [.333, .666], inplace = True)
    numberOfObservations = dataFrame.shape[0]
    dimensionsUsed       = list(dataFrame.keys())[dimensionIndexStart : dimensionIndexEnd]
    numberOfDimensions   = len(dimensionsUsed)

    observationMatrix    = numpy.zeros((numberOfObservations, numberOfDimensions))
    for i in range(numberOfDimensions):
        observationMatrix[ : , i] = dataFrame.get(dimensionsUsed[i])

    return observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions



def loadAvailabilityMatrix(observationMatrix):
    '''
    Load the availability matrix from 'observationMatrix'.
    'True' means available; 'False' means unavailable.

    observationMatrix: numpy.ndarray of floats, shape '(numberOfObservations, numberOfDimensions)'
    '''

    return numpy.logical_not(numpy.isnan(observationMatrix))



def distancesGower(observationMatrix, hasObservation, dimensionalWeights, clusterA, clusterB):
    '''
    Calculate Gower's distance for all pairs of observations, where the first observation is from 'clusterA' and the second from 'clusterB'.
    '''

    # Fetch cluster A.
    observationsClusterA           = observationMatrix[clusterA]   # shape: '(numberOfObservationsClusterA, numberOfDimensions)'
    hasObservationClusterA         = hasObservation[clusterA]      # shape: '(numberOfObservationsClusterA, numberOfDimensions)'
    numberOfObservationsClusterA   = observationsClusterA.shape[0] # in 1

    # Fetch cluster B.
    observationsClusterB           = observationMatrix[clusterB]   # shape: '(numberOfObservationsClusterB, numberOfDimensions)'
    hasObservationClusterB         = hasObservation[clusterB]      # shape: '(numberOfObservationsClusterB, numberOfDimensions)'
    numberOfObservationsClusterB   = observationsClusterB.shape[0] # in 1

    # Generate matrices of shape: '(numberOfObservationsClusterA * numberOfObservationsClusterB, numberOfDimensions)'
    observationsClusterAExtended   = numpy.repeat(observationsClusterA,   numberOfObservationsClusterB, axis = 0)
    hasObservationClusterAExtended = numpy.repeat(hasObservationClusterA, numberOfObservationsClusterB, axis = 0)
    observationsClusterBExtended   = numpy.tile(observationsClusterB,     (numberOfObservationsClusterA, 1))
    hasObservationClusterBExtended = numpy.tile(hasObservationClusterB,   (numberOfObservationsClusterA, 1))
    hasObservationProduct          = hasObservationClusterAExtended * hasObservationClusterBExtended

    # Calculate Gower's distances.
    distances                      = numpy.matmul(numpy.abs(observationsClusterAExtended - observationsClusterBExtended) * hasObservationProduct, dimensionalWeights) / numpy.matmul(hasObservationProduct, dimensionalWeights) # shape: '(numberOfObservationsClusterA * numberOfObservationsClusterB, )'

    return distances



def silhouettes(observationMatrix, hasObservation, dimensionalWeights, clusterList):
    '''
    Returns, given a list of all clusters, the observations and their availability, and a weighting scheme for the dimensions,
    the silhouette (or score) of each observation.
    The silhouette for observation i is defined (b(i) - a(i)) / max(a(i), b(i)), but whenever an observation is the sole element of its cluster,
    the corresponding silhouette is 0.
    '''

    numberOfObservations = observationMatrix.shape[0]
    numberOfClusters     = len(clusterList)

    # Store, for each observation, the value of a, b and the index and size of its cluster.
    arrayA               = numpy.zeros(numberOfObservations, dtype = numpy.float64)
    arrayB               = numpy.zeros(numberOfObservations, dtype = numpy.float64)
    arrayClusterIndex    = numpy.empty(numberOfObservations, dtype = numpy.uint8)
    arrayClusterSize     = numpy.empty(numberOfObservations, dtype = numpy.uint8)


    # Calculate 'a' for each observation.
    for indexCluster in range(numberOfClusters):
        cluster                     = clusterList[indexCluster]
        numberOfObservationsCluster = len(cluster)
        arrayClusterIndex[cluster]  = indexCluster
        arrayClusterSize[cluster]   = numberOfObservationsCluster

        if (numberOfObservationsCluster > 1):
            distances       = distancesGower(observationMatrix, hasObservation, dimensionalWeights, cluster, cluster).reshape((numberOfObservationsCluster, numberOfObservationsCluster))
            numpy.fill_diagonal(distances, numpy.nan)

            arrayA[cluster] = numpy.nanmean(distances, axis = 0)


    # Calculate 'b' for each observation.
    for indexObservation in range(numberOfObservations):
        indexClusterOwn     = arrayClusterIndex[indexObservation]
        distanceMeanMinimum = None

        for indexCluster in range(numberOfClusters):
            if (indexCluster != indexClusterOwn):
                distances    = distancesGower(observationMatrix, hasObservation, dimensionalWeights, [indexObservation], clusterList[indexCluster])
                distanceMean = numpy.mean(distances)

                if (distanceMeanMinimum == None or distanceMean < distanceMeanMinimum):
                    distanceMeanMinimum = distanceMean

        arrayB[indexObservation] = distanceMeanMinimum


    arraySilhouettes     = (arrayB - arrayA) / numpy.maximum(arrayA, arrayB) * numpy.greater(arrayClusterSize, 1)
    return arraySilhouettes



def clusterMean(observationMatrixCluster, hasObservationCluster):
    '''
    Missing values in 'observationMatrixCluster' are represented by -1.
    Missing values in the final version of 'observationMatrixClusterMean' are also represented by -1.
    '''
    hasObservationClusterSum         = numpy.sum(hasObservationCluster, axis = 0)
    observationMatrixClusterMean     = numpy.sum(observationMatrixCluster * hasObservationCluster, axis = 0) / numpy.maximum(hasObservationClusterSum, 1) # missing: 0
    hasObservationClusterMean        = (hasObservationClusterSum > 0)
    observationMatrixClusterMean[numpy.logical_not(hasObservationClusterMean)] = -1 # missing: -1

    return (observationMatrixClusterMean, hasObservationClusterMean)



def distancesToClusterMean(observationMatrixCluster, hasObservationCluster, dimensionalWeights):
    '''
    '''
    observationMatrixClusterMean, hasObservationClusterMean = clusterMean(observationMatrixCluster, hasObservationCluster)

    observationMatrixClusterAppended = numpy.concatenate((observationMatrixCluster, observationMatrixClusterMean[None, : ]))
    hasObservationClusterAppended    = numpy.concatenate((hasObservationCluster, hasObservationClusterMean[None, : ]))

    clusterSize                      = observationMatrixCluster.shape[0]
    distances                        = distancesGower(observationMatrixClusterAppended, hasObservationClusterAppended, dimensionalWeights, range(clusterSize), [clusterSize])

    return distances
