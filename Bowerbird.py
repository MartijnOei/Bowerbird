'''
Martijn Simon Soen Liong Oei, April 12021 H.E.
'''




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.
'''

import numpy, pandas


def loadObservationMatrix(filePath, dimensionIndexStart, dimensionIndexEnd, sep = ";"):
    '''
    Load the observation matrix from a .csv-file at 'filePath',
    including only columns indexed by 'dimensionIndexStart' up to (but not including) 'dimensionIndexEnd'.

    observationMatrix: numpy.ndarray of floats, shape  '(numberOfObservations, numberOfDimensions)'
    dimensionsUsed:    list of strings,         length 'numberOfDimensions'
    '''
    # Load file.
    dataFrame            = pandas.read_csv(filePath, sep = sep)
    numberOfObservations = dataFrame.shape[0] # in 1

    # This line is a relic specific to the narcolepsy project:
    dataFrame.replace(["1/3", "2/3"], [.333, .666], inplace = True)

    # Fetch dimension names.
    if (dimensionIndexStart is None):
        dimensionIndexStart = 0
    if (dimensionIndexEnd is None):
        dimensionsUsed = list(dataFrame.keys())[dimensionIndexStart : ]
    else:
        dimensionsUsed = list(dataFrame.keys())[dimensionIndexStart : dimensionIndexEnd]
    numberOfDimensions   = len(dimensionsUsed) # in 1

    # Fetch data.
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




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

Prepare full data set.
'''

import numpy, os

def AHCPrepareDataSetFull(directoryData, fileName, indexColumnStart = None, indexColumnEnd = None, dataSetName = "full", normalise = None, normalisePercentile = 2):
    '''
    '''
    # Load data set.
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)

    # Either 'normalise' is 'None', or it is a list containing 'True' or 'False' for each column.
    if (not (normalise is None)):
        for indexDimension in range(numberOfDimensions):
            observationsDimension = observationMatrix[ : , indexDimension]
            if (normalise[indexDimension]):
                clipMin = numpy.percentile(observationsDimension, normalisePercentile)
                clipMax = numpy.percentile(observationsDimension, 100 - normalisePercentile)

                observationMatrix[ : , indexDimension] = (numpy.clip(observationsDimension, clipMin, clipMax) - clipMin) / (clipMax - clipMin)


    # Save data set.
    directoryDataSet = directoryData + dataSetName + "/"
    if (not os.path.exists(directoryDataSet)):
        os.makedirs(directoryDataSet)
    numpy.save(directoryDataSet + "observationMatrix.npy", observationMatrix)




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

This program performs agglomerative hierarchical clustering using Gower's distance.
The data consists of 'numberOfObservations' vectors of length 'numberOfDimensions', and is allowed to have missing entries.
'''

import numpy, time

def matrixNew(distanceInterClustersMatrix, numberOfClusters, indexClusterA, indexClusterB):
    '''
    Construct the new version of 'distanceInterClustersMatrix' (except for the last column, which is calculated later).
    '''

    matrixNew = numpy.full((numberOfClusters - 1, numberOfClusters - 1), numpy.inf)

    matrixNew[ : indexClusterA, : indexClusterA]                                                  = distanceInterClustersMatrix[ : indexClusterA, : indexClusterA]
    matrixNew[ : indexClusterA, indexClusterA : indexClusterB - 1]                                = distanceInterClustersMatrix[ : indexClusterA, indexClusterA + 1 : indexClusterB]
    matrixNew[ : indexClusterA, indexClusterB - 1 : numberOfClusters - 2]                         = distanceInterClustersMatrix[ : indexClusterA, indexClusterB + 1 : ]
    matrixNew[indexClusterA : indexClusterB - 1, indexClusterA : indexClusterB - 1]               = distanceInterClustersMatrix[indexClusterA + 1 : indexClusterB, indexClusterA + 1 : indexClusterB]
    matrixNew[indexClusterA : indexClusterB - 1, indexClusterB - 1 : numberOfClusters - 2]        = distanceInterClustersMatrix[indexClusterA + 1 : indexClusterB, indexClusterB + 1 : ]
    matrixNew[indexClusterB - 1 : numberOfClusters - 2, indexClusterB - 1 : numberOfClusters - 2] = distanceInterClustersMatrix[indexClusterB + 1 : , indexClusterB + 1 : ]
    return matrixNew



def AHCCompute(directoryData,
               linkageType,                 # all options: "complete", "average", "single"
               dimensionalWeights,          # Gower's distance weight factors
               numberOfClustersStartSaving, # the highest number of clusters for which data is saved (... and the lowest being fixed to 2)
               dataSetName = "full"
               ):
    '''
    '''

    # Load data.
    observationMatrix                   = numpy.load(directoryData + dataSetName + "/observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape
    hasObservation                      = loadAvailabilityMatrix(observationMatrix)
    observationMatrix[numpy.isnan(observationMatrix)] = -1 # after 'hasObservation' has been created, we set missing entries to -1 to ensure NumPy routines work correctly

    print ("Finished loading data!")
    print ("Number of observations:", numberOfObservations)
    print ("Number of dimensions:",   numberOfDimensions)


    # Initialise the cluster list: every cluster contains one point.
    clusterList                         = []
    for i in range(numberOfObservations):
        clusterList.append([i])
    numberOfClusters                    = numberOfObservations


    # Initialise the intra-cluster distance list.
    # Each element is a NumPy array with all distances between the observations of a particular cluster.
    distanceIntraClustersList           = []
    for i in range(numberOfObservations):
        distanceIntraClustersList.append(numpy.array([]))


    # Calculate the inter-cluster distance matrix. Initially, this matrix is the same for all linkage types.
    timeStart                           = time.time()

    distanceInterClustersMatrix         = distancesGower(observationMatrix, hasObservation, dimensionalWeights, range(numberOfObservations), range(numberOfObservations)).reshape((numberOfObservations, numberOfObservations))
    distanceInterClustersMatrix[numpy.tril_indices(numberOfClusters)] = numpy.inf

    distanceInterClustersMatrixComplete = numpy.copy(distanceInterClustersMatrix)
    distanceInterClustersMatrixAverage  = numpy.copy(distanceInterClustersMatrix)
    distanceInterClustersMatrixSingle   = numpy.copy(distanceInterClustersMatrix)

    timeEnd                             = time.time()
    print("Time needed to initialise inter-cluster distance matrix (s):", numpy.round(timeEnd - timeStart, 3))


    # Initialise dendrogram information.
    dendrogramMatrix                    = numpy.empty((numberOfObservations - 1, 4), dtype = numpy.float64)
    clusterIndicesList                  = list(range(numberOfObservations)) # In Python 3, the 'range'-function doesn't produce a list, but got its own type.
    iterationIndex                      = 0


    # Execute hierarchical clustering.
    while (numberOfClusters > 2):

        # Find the indices of the clusters that must be merged. By construction, 'indexClusterA' < 'indexClusterB'.
        if   (linkageType == "complete"):
            indexFlattened = numpy.argmin(distanceInterClustersMatrixComplete)
        elif (linkageType == "average"):
            indexFlattened = numpy.argmin(distanceInterClustersMatrixAverage)
        elif (linkageType == "single"):
            indexFlattened = numpy.argmin(distanceInterClustersMatrixSingle)

        indexClusterA                       = indexFlattened // numberOfClusters
        indexClusterB                       = indexFlattened % numberOfClusters

        # Create proto-versions of the new inter-cluster distance matrices.
        matrixNewComplete                   = matrixNew(distanceInterClustersMatrixComplete, numberOfClusters, indexClusterA, indexClusterB)
        matrixNewAverage                    = matrixNew(distanceInterClustersMatrixAverage,  numberOfClusters, indexClusterA, indexClusterB)
        matrixNewSingle                     = matrixNew(distanceInterClustersMatrixSingle,   numberOfClusters, indexClusterA, indexClusterB)

        # Form the new cluster by merging two old ones.
        clusterA                            = clusterList[indexClusterA]
        clusterB                            = clusterList[indexClusterB]
        clusterNew                          = clusterA + clusterB

        # Update the list of clusters. We first remove the cluster with the highest index, to ensure that the second command removes the correct cluster.
        clusterList.pop(indexClusterB)
        clusterList.pop(indexClusterA)
        clusterList.append(clusterNew)

        # Find the index of the new cluster.
        indexClusterNew                     = numberOfClusters - 2

        # Fill the last column of the proto-versions of the new inter-cluster distance matrices iteratively.
        for indexClusterOld in range(indexClusterNew):
            clusterOld                    = clusterList[indexClusterOld]
            distances                     = distancesGower(observationMatrix, hasObservation, dimensionalWeights, clusterOld, clusterNew)

            # Determine the distance between the clusters under the linkage schemes.
            distanceInterClustersComplete = numpy.amax(distances)
            distanceInterClustersAverage  = numpy.mean(distances)
            distanceInterClustersSingle   = numpy.amin(distances)

            # Fill the right entries with these 3 distances.
            matrixNewComplete[indexClusterOld, indexClusterNew] = distanceInterClustersComplete
            matrixNewAverage [indexClusterOld, indexClusterNew] = distanceInterClustersAverage
            matrixNewSingle  [indexClusterOld, indexClusterNew] = distanceInterClustersSingle

        # Now the newly-built matrices are ready, update the 3 inter-cluster distance matrices.
        distanceInterClustersMatrixComplete = matrixNewComplete
        distanceInterClustersMatrixAverage  = matrixNewAverage
        distanceInterClustersMatrixSingle   = matrixNewSingle

        # Update the dendrogram information.
        dendrogramMatrix[iterationIndex, 0] = clusterIndicesList[indexClusterB]
        dendrogramMatrix[iterationIndex, 1] = clusterIndicesList[indexClusterA]
        dendrogramMatrix[iterationIndex, 2] = 1#distanceInterClustersSmallest
        dendrogramMatrix[iterationIndex, 3] = len(clusterNew)
        clusterIndicesList.pop(indexClusterB)
        clusterIndicesList.pop(indexClusterA)
        clusterIndicesList.append(numberOfObservations + iterationIndex)
        iterationIndex                     += 1


        # Update the number of clusters.
        numberOfClusters                   -= 1


        # Update the list of intra-cluster distance arrays.
        # First, find the array with all distances within the new cluster.
        distanceIntraClusterNew             = numpy.concatenate((distanceIntraClustersList[indexClusterA], distanceIntraClustersList[indexClusterB], distancesGower(observationMatrix, hasObservation, dimensionalWeights, clusterA, clusterB)))
        # Then update 'distanceIntraClustersList'.
        distanceIntraClustersList.pop(indexClusterB)
        distanceIntraClustersList.pop(indexClusterA)
        distanceIntraClustersList.append(distanceIntraClusterNew)


        # Calculate the metrics, if we save results for this number of clusters.
        if (numberOfClusters <= numberOfClustersStartSaving):
            # Calculate mean inter-cluster distances.
            indices                           = numpy.triu_indices(numberOfClusters, 1)
            distanceInterClustersMeanComplete = numpy.mean(distanceInterClustersMatrixComplete[indices])
            distanceInterClustersMeanAverage  = numpy.mean(distanceInterClustersMatrixAverage [indices])
            distanceInterClustersMeanSingle   = numpy.mean(distanceInterClustersMatrixSingle  [indices])

            # Calculate mean intra-cluster distance.
            distanceIntraClustersMean         = numpy.mean(numpy.concatenate(distanceIntraClustersList))

            # Calculate silhouettes.
            silhouettesAll                    = silhouettes(observationMatrix, hasObservation, dimensionalWeights, clusterList)
            silhouettesMean                   = numpy.mean(silhouettesAll)
            silhouettesSD                     = numpy.std(silhouettesAll)

            # Calculate Dunn's indices.
            # Because we take the minimum, the occurrence of 'numpy.inf' in the array does not matter, and we do not have to slice the arrays first.
            distanceInterClustersMinComplete  = numpy.amin(distanceInterClustersMatrixComplete)
            distanceInterClustersMinAverage   = numpy.amin(distanceInterClustersMatrixAverage)
            distanceInterClustersMinSingle    = numpy.amin(distanceInterClustersMatrixSingle)
            diameterMaximum                   = numpy.amax(numpy.concatenate(distanceIntraClustersList))
            indexDunnComplete                 = distanceInterClustersMinComplete / diameterMaximum
            indexDunnAverage                  = distanceInterClustersMinAverage  / diameterMaximum
            indexDunnSingle                   = distanceInterClustersMinSingle   / diameterMaximum

            # Calculate coefficient of determination.
            varianceModelCurrent              = 0
            for i in range(numberOfClusters):
                distances             = distancesToClusterMean(observationMatrix[clusterList[i]], hasObservation[clusterList[i]], dimensionalWeights)
                varianceModelCurrent += numpy.sum(numpy.square(distances))

            distances                         = distancesToClusterMean(observationMatrix, hasObservation, dimensionalWeights)
            varianceModelTrivial              = numpy.sum(numpy.square(distances))
            coefficientOfDetermination        = 1 - varianceModelCurrent / varianceModelTrivial


            # Combine all single-number metrics in one array.
            metricsSingleNumber               = numpy.array([distanceInterClustersMeanComplete, distanceInterClustersMeanAverage, distanceInterClustersMeanSingle, distanceIntraClustersMean, silhouettesMean, silhouettesSD, distanceInterClustersMinComplete, distanceInterClustersMinAverage, distanceInterClustersMinSingle, diameterMaximum, indexDunnComplete, indexDunnAverage, indexDunnSingle, coefficientOfDetermination])


            # Save results.
            linkageTypeCapitalised            = linkageType.capitalize()
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList"                         + ".npy", clusterList)
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "DistanceInterClustersMatrixComplete" + ".npy", distanceInterClustersMatrixComplete)
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "DistanceInterClustersMatrixAverage"  + ".npy", distanceInterClustersMatrixAverage)
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "DistanceInterClustersMatrixSingle"   + ".npy", distanceInterClustersMatrixSingle)
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "DistanceIntraClustersList"           + ".npy", distanceIntraClustersList)
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "SilhouettesAll"                      + ".npy", silhouettesAll)
            numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "MetricsSingleNumber"                 + ".npy", metricsSingleNumber)

        print ("Finished iteration. Number of clusters left (1):", numberOfClusters)


    dendrogramMatrix[numberOfObservations - 2, 0] = clusterIndicesList[0]
    dendrogramMatrix[numberOfObservations - 2, 1] = clusterIndicesList[1]
    dendrogramMatrix[numberOfObservations - 2, 2] = 1
    dendrogramMatrix[numberOfObservations - 2, 3] = numberOfObservations
    numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + "DendrogramMatrix.npy", dendrogramMatrix)




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

This program visualises the agglomerative hierarchical clustering results generated by 'AHCCompute'.
Amongst other things, it plots the evolution of outcome metrics through the iterations. Use this to determine a number of clusters to stop clustering at.
'''

from matplotlib import cm, colorbar, gridspec, pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster import hierarchy
import copy, numpy, os
import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["savefig.dpi"] = 400


def AHCResultsVisualisation(directoryData,                                          # path to the main data    directory
                            directoryFigures,                                       # path to the main figures directory
                            linkageType,                                            # determines the results to show
                            numberOfClustersHighest,                                # highest cluster number in progression plots
                            numberOfClustersLowest,                                 # lowest  cluster number in progression plots (and number in single-cluster plots)
                            dimensionsUsed,                                         # names   of the dimensions
                            listGroupNames,                                         # names   of the dimension groups
                            listGroupIndices,                                       # indices of the dimension groups
                            dataSetName                             = "full",       # name of the data set (a folder with this name should exist in 'directoryData')
                            colourMapSilhouettesAll                 = cm.RdBu,
                            silhouetteColourMin                     = -.75,         # silhouette value corresponding to left  extreme of colour map
                            silhouetteColourMax                     = .75,          # silhouette value corresponding to right extreme of colour map
                            radius                                  = .75,          # silhouette value radius shown
                            colourMapClusters                       = cm.RdBu_r,    # colour map for the observation matrix and the individual clusters
                            colourMapClusterMeans                   = cm.Greens,    # colour map for background of dimension names and dimension group names
                            rowSpanGroups                           = 2,
                            rowSpanDimensions                       = 10,
                            rowSpanCluster                          = 5,
                            rowSpanWhiteSpace                       = 1,
                            rowSpanColourBar                        = 1,
                            colourMapDistanceInterClustersMatrices  = cm.cividis_r, # colour map for inter-cluster distance matrices
                            colourBarWidth                          = "2.5%",
                            colourBarDistance                       = 0.06,
                            distanceVMin                            = 50,           # in percentage points of the maximum distance
                            fontSizeGroups                          = 12,
                            fontSizeDimensions                      = 12,
                            fontSizeClusters                        = 12,
                            plotCoefficientsOfDetermination         = True,
                            plotSilhouettesAll                      = True,
                            plotSilhouettesSummary                  = True,
                            plotIndicesDunn                         = True,
                            plotDistanceMeans                       = True,
                            plotClusters                            = True,
                            plotClusterMeans                        = True,
                            plotDistanceInterClustersMatrices       = True,
                            plotDendrogram                          = True,
                            figureWidthCoefficientsOfDetermination  = 6,           # in inch
                            figureHeightCoefficientsOfDetermination = 3,           # in inch
                            figureWidthSilhouettesAll               = 10,          # in inch
                            figureHeightSilhouettesAll              = 4,           # in inch
                            figureWidthSilhouettesSummary           = 6,           # in inch
                            figureHeightSilhouettesSummary          = 3,           # in inch
                            figureWidthIndicesDunn                  = 6,           # in inch
                            figureHeightIndicesDunn                 = 3,           # in inch
                            figureWidthDistanceMeans                = 8,           # in inch
                            figureHeightDistanceMeans               = 5,           # in inch
                            figureWidthClustersColourBar            = .35,         # in inch; determines colour bar width
                            figureHeightClustersBarCode             = 1.2,         # in inch; determines bar code height
                            figureWidthClusterMeans                 = 10,          # in inch
                            figureHeightClusterMeans                = 8,           # in inch
                            figureWidthDistanceInterClustersMatrix  = 6,           # in inch
                            figureHeightDistanceInterClustersMatrix = 6,           # in inch
                            figureWidthDendrogram                   = 6,           # in inch
                            figureHeightDendrogram                  = 4,           # in inch
                            leftClusterMeans                        = .08,         # in 1 #.04,
                            rightClusterMeans                       = .985,        # in 1
                            bottomClusterMeans                      = .04,         # in 1 #.02,
                            topClusterMeans                         = .99,         # in 1
                            plotClusterMeansText                    = True,
                            colourDendrogram                        = "crimson",   # colour of the dendrogram
                            figureExtension                         = ".pdf"
                            ):
    '''
    Visualises agglomerative hierarchical clustering results.
    '''

    colourMapClusters = copy.copy(cm.get_cmap(colourMapClusters))
    colourMapClusters.set_bad(color = "white")

    # Load 'observationMatrix'.
    observationMatrix                   = numpy.load(directoryData + dataSetName + "/observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape
    observationNames                    = numpy.arange(numberOfObservations)

    # Load data.
    linkageTypeCapitalised              = linkageType.capitalize()
    clusterList                         = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "ClusterList"    + ".npy", allow_pickle = True)
    silhouettesAll                      = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "SilhouettesAll" + ".npy")
    distanceInterClustersMatrixComplete = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "DistanceInterClustersMatrixComplete" + ".npy")
    distanceInterClustersMatrixAverage  = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "DistanceInterClustersMatrixAverage" + ".npy")
    distanceInterClustersMatrixSingle   = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "DistanceInterClustersMatrixSingle" + ".npy")

    numberOfData                        = numberOfClustersHighest - numberOfClustersLowest + 1 # in 1
    distanceInterClustersMeansComplete  = numpy.empty(numberOfData)
    distanceInterClustersMeansAverage   = numpy.empty(numberOfData)
    distanceInterClustersMeansSingle    = numpy.empty(numberOfData)
    distanceIntraClustersMeans          = numpy.empty(numberOfData)
    silhouetteMeans                     = numpy.empty(numberOfData)
    silhouetteSDs                       = numpy.empty(numberOfData)
    indicesDunnComplete                 = numpy.empty(numberOfData)
    indicesDunnAverage                  = numpy.empty(numberOfData)
    indicesDunnSingle                   = numpy.empty(numberOfData)
    coefficientsOfDetermination         = numpy.empty(numberOfData) # in 1

    for numberOfClusters, i in zip(range(numberOfClustersLowest, numberOfClustersHighest + 1)[::-1], range(numberOfData)):
        metricsSingleNumber                   = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "MetricsSingleNumber" + ".npy")
        distanceInterClustersMeansComplete[i] = metricsSingleNumber[0]
        distanceInterClustersMeansAverage [i] = metricsSingleNumber[1]
        distanceInterClustersMeansSingle  [i] = metricsSingleNumber[2]
        distanceIntraClustersMeans        [i] = metricsSingleNumber[3]
        silhouetteMeans                   [i] = metricsSingleNumber[4]
        silhouetteSDs                     [i] = metricsSingleNumber[5]
        indicesDunnComplete               [i] = metricsSingleNumber[10]
        indicesDunnAverage                [i] = metricsSingleNumber[11]
        indicesDunnSingle                 [i] = metricsSingleNumber[12]
        coefficientsOfDetermination       [i] = metricsSingleNumber[13]

    rangeClusterNumberMetrics           = numpy.arange(numberOfClustersLowest, numberOfClustersHighest + 1)[ : : -1]


    if (plotClusters or plotClusterMeans):
        clusterSizeList             = []
        clusterObservationNamesList = []
        clusterMatrixPlotList       = []
        clusterMeansList            = []

        for i in range(numberOfClustersLowest):
            clusterObservationIndices = clusterList[i]
            clusterSize               = len(clusterObservationIndices)
            clusterObservationNames   = observationNames[clusterObservationIndices]
            clusterMatrix             = observationMatrix[clusterObservationIndices]
            clusterMatrixPlot         = clusterMatrix.astype(numpy.float64)
            clusterMatrixPlot[(clusterMatrix == -1)] = numpy.nan
            clusterMeans              = numpy.nanmean(clusterMatrixPlot, axis = 0)

            # Save quantities that can be used in plots later.
            clusterSizeList.append(clusterSize)
            clusterObservationNamesList.append(clusterObservationNames)
            clusterMatrixPlotList.append(clusterMatrixPlot)
            clusterMeansList.append(clusterMeans)


    # Generate the directory for the figures, if it does not exist yet.
    if (not os.path.exists(directoryFigures + dataSetName + "/")):
        os.makedirs(directoryFigures + dataSetName + "/")



    if (plotCoefficientsOfDetermination):
        pyplot.figure(figsize = (figureWidthCoefficientsOfDetermination, figureHeightCoefficientsOfDetermination))
        pyplot.scatter(rangeClusterNumberMetrics, coefficientsOfDetermination * 100, s = 6, c = "crimson")
        pyplot.plot(rangeClusterNumberMetrics, coefficientsOfDetermination * 100, ls = "--", c = "crimson", alpha = .5)
        pyplot.gca().invert_xaxis()
        pyplot.grid(ls = "--", alpha = .2)
        pyplot.xticks(rangeClusterNumberMetrics)
        pyplot.xlim(rangeClusterNumberMetrics[0] + .1, rangeClusterNumberMetrics[-1] - .1)
        #pyplot.xlim(rangeClusterNumberMetrics[0], rangeClusterNumberMetrics[-1])
        pyplot.xlabel("number of clusters (1)")
        pyplot.ylabel(r"fraction of explained variance (\%)")
        pyplot.title(r"goodness of fit: coefficient of determination $R^2$ $\vert$ " + r"\textbf{" + linkageType + r"}" + " linkage")
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "CoefficientsOfDetermination" + figureExtension)
        pyplot.close()



    if (plotSilhouettesAll):

        pyplot.figure(figsize = (figureWidthSilhouettesAll, figureHeightSilhouettesAll))
        for indexCluster in range(numberOfClustersLowest):
            clusterObservationIndices = clusterList[indexCluster]
            clusterSize               = len(clusterObservationIndices)
            clusterSilhouettesSorted  = numpy.sort(silhouettesAll[clusterObservationIndices])
            clusterSilhouetteMean     = numpy.mean(clusterSilhouettesSorted)
            clusterSilhouetteSD       = numpy.std( clusterSilhouettesSorted)

            axesCluster               = pyplot.subplot2grid((1, numberOfClustersLowest), (0, indexCluster), rowspan = 1, colspan = 1)

            axesCluster.bar(range(clusterSize), clusterSilhouettesSorted, width = 1, color = colourMapSilhouettesAll(numpy.clip((clusterSilhouettesSorted - silhouetteColourMin) / (silhouetteColourMax - silhouetteColourMin), 0, 1)), label = "mean: " + str(numpy.round(clusterSilhouetteMean, 2)) + "\nSD:\ \ \ \ " + str(numpy.round(clusterSilhouetteSD, 2)))
            axesCluster.set_xlim(-0.5, clusterSize - 0.5)
            axesCluster.set_ylim(-radius, radius)
            axesCluster.set_xticks([])

            axesCluster.grid(True, ls = "--", alpha = .3)

            if (indexCluster == 0):
                axesCluster.set_ylabel("silhouette (1)", fontsize = "large")
                axesCluster.yaxis.set_tick_params(labelsize = "large")
            else:
                #axesCluster.set_yticks([])
                #for tick in axesCluster.yaxis.get_major_ticks():
                #    tick.tick1On = False
                #    tick.tick2On = False
                #pyplot.setp(axesCluster.get_yticks(), visible = False)
                axesCluster.yaxis.set_tick_params(length = 0)#label1On = False, label2On = False,
                axesCluster.set_yticklabels([])

            axesCluster.set_title("cluster " + str(indexCluster + 1), fontsize = "large")# + "\nmean: " + str(numpy.round(clusterSilhouetteMean, 2)), fontsize = "large")
            #axesCluster.set_title("cluster " + str(indexCluster + 1) + "\n" + "$\mu = " + str(numpy.round(clusterSilhouetteMean, 2)) + "$", fontsize = "large")
            #axesCluster.set_title(r"\textbf{" + str(indexCluster + 1) + r"}:" + "\n" + r"$N = " + str(clusterSize) + r"$ $\vert$ $\mu = " + str(numpy.round(clusterSilhouetteMean, 2)) + "$")
            axesCluster.legend(loc = "lower right")
        #pyplot.suptitle(r"\textbf{all clusters}: $N = " + str(numberOfObservations) + r"$ $\vert$ $\mu = " + str(numpy.round(numpy.mean(silhouettesAll), 2)) + "$")
        #pyplot.subplots_adjust(left = 0.07, right = .99, bottom = .01, top = .85, wspace = 0)
        #pyplot.yticks(fontsize = "large")
        pyplot.subplots_adjust(left = .08, right = .99, bottom = .02, top = .9, wspace = .02)
        #pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "SilhouettesAll" + figureExtension)
        pyplot.close()



    if (plotSilhouettesSummary):
        pyplot.figure(figsize = (figureWidthSilhouettesSummary, figureHeightSilhouettesSummary))
        pyplot.scatter(rangeClusterNumberMetrics, silhouetteMeans, s = 6, label = r"mean", c = "crimson", zorder = 2)#$\mu$
        pyplot.plot(   rangeClusterNumberMetrics, silhouetteMeans, ls = "--", alpha = .5, c = "crimson", zorder = 2)
        pyplot.axhline(y = 0, c = "gray", alpha = .5, zorder = 1)

        #pyplot.fill_between(rangeClusterNumberMetrics, silhouetteMeans - 2 * silhouetteSDs, silhouetteMeans + 2 * silhouetteSDs, alpha = .2, color = "orangered", lw = 0)
        pyplot.fill_between(rangeClusterNumberMetrics, silhouetteMeans - 1 * silhouetteSDs, silhouetteMeans + 1 * silhouetteSDs, alpha = .2, color = "crimson", lw = 0, label = r"$\pm1$ SD interval", zorder = 2)#\sigma

        pyplot.grid(ls = "--", alpha = .25)
        pyplot.gca().invert_xaxis()

        pyplot.xticks(rangeClusterNumberMetrics)
        pyplot.xlim(numberOfClustersHighest, numberOfClustersLowest)
        #pyplot.xlim(numberOfClustersHighest + 0.1, numberOfClustersLowest - 0.1)
        #pyplot.ylim(-1, 1)

        pyplot.xlabel("number of clusters (1)")
        pyplot.ylabel("silhouette (1)")
        pyplot.legend(loc = "upper left")
        pyplot.title(r"silhouette summary statistics $\vert$ " + r"\textbf{" + linkageType + r"}" + " linkage") #during agglomerative hierarchical clustering (mean with $\pm$1 standard deviation interval)
        #pyplot.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.08, top = 0.92)
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "ProgressionSilhouettes" + figureExtension)
        pyplot.close()



    if (plotIndicesDunn):
        pyplot.figure(figsize = (figureWidthIndicesDunn, figureHeightIndicesDunn))
        pyplot.scatter(rangeClusterNumberMetrics, indicesDunnComplete / indicesDunnComplete[0], label = "complete", s = 4, c = "crimson")
        pyplot.plot(   rangeClusterNumberMetrics, indicesDunnComplete / indicesDunnComplete[0], ls = "--", alpha = .5,     c = "crimson")
        pyplot.scatter(rangeClusterNumberMetrics, indicesDunnAverage  / indicesDunnAverage[0],  label = "average",  s = 4, c = "yellowgreen")
        pyplot.plot(   rangeClusterNumberMetrics, indicesDunnAverage  / indicesDunnAverage[0],  ls = "--", alpha = .5,     c = "yellowgreen")
        pyplot.scatter(rangeClusterNumberMetrics, indicesDunnSingle   / indicesDunnSingle[0],   label = "single",   s = 4, c = "navy")
        pyplot.plot(   rangeClusterNumberMetrics, indicesDunnSingle   / indicesDunnSingle[0],   ls = "--", alpha = .5,     c = "navy")

        pyplot.grid(ls = "--", alpha = .25)
        pyplot.gca().invert_xaxis()
        pyplot.xticks(rangeClusterNumberMetrics)
        pyplot.xlim(numberOfClustersHighest + 0.1, numberOfClustersLowest - 0.1)

        pyplot.xlabel("number of clusters (1)")
        pyplot.ylabel("relative Dunn's index (1)")

        pyplot.legend(loc = "upper left")

        pyplot.title(r"Dunn's indices $\vert$ " + r"\textbf{" + linkageType + r"}" + " linkage") #during agglomerative hierarchical clustering
        #pyplot.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.08, top = 0.92)
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "ProgressionDunnsIndices" + figureExtension)
        pyplot.close()



    if (plotDistanceMeans):
        pyplot.figure(figsize = (figureWidthDistanceMeans, figureHeightDistanceMeans))
        gridspec.GridSpec(10, 1)
        axesMeans          = pyplot.subplot2grid((10, 1), (0, 0), rowspan = 7, colspan = 1)
        axesRatio          = pyplot.subplot2grid((10, 1), (7, 0), rowspan = 3, colspan = 1)

        ratioComplete      = distanceInterClustersMeansComplete / distanceIntraClustersMeans
        ratioAverage       = distanceInterClustersMeansAverage  / distanceIntraClustersMeans
        ratioSingle        = distanceInterClustersMeansSingle   / distanceIntraClustersMeans

        axesMeans.scatter(rangeClusterNumberMetrics, distanceInterClustersMeansComplete, s = 4, label = "inter-cluster mean (complete)", c = "crimson")
        axesMeans.plot(   rangeClusterNumberMetrics, distanceInterClustersMeansComplete, ls = "--", alpha = .5, c = "crimson")
        axesMeans.scatter(rangeClusterNumberMetrics, distanceInterClustersMeansAverage,  s = 4, label = "inter-cluster mean (average)", c = "orangered")
        axesMeans.plot(   rangeClusterNumberMetrics, distanceInterClustersMeansAverage,  ls = "--", alpha = .5, c = "orangered")
        axesMeans.scatter(rangeClusterNumberMetrics, distanceInterClustersMeansSingle,   s = 4, label = "inter-cluster mean (single)", c = "goldenrod")
        axesMeans.plot(   rangeClusterNumberMetrics, distanceInterClustersMeansSingle,   ls = "--", alpha = .5, c = "goldenrod")

        axesMeans.scatter(rangeClusterNumberMetrics, distanceIntraClustersMeans,         s = 4, label = "intra-cluster mean", c = "yellowgreen")
        axesMeans.plot(   rangeClusterNumberMetrics, distanceIntraClustersMeans,         ls = "--", alpha = .5, c = "yellowgreen")

        axesRatio.scatter(rangeClusterNumberMetrics, ratioComplete,                      s = 4, label = "inter- over intra-cluster mean (complete)", c = "mediumseagreen")
        axesRatio.plot(   rangeClusterNumberMetrics, ratioComplete,                      ls = "--", alpha = .5, c = "mediumseagreen")
        axesRatio.scatter(rangeClusterNumberMetrics, ratioAverage,                       s = 4, label = "inter- over intra-cluster mean (average)", c = "cornflowerblue")
        axesRatio.plot(   rangeClusterNumberMetrics, ratioAverage,                       ls = "--", alpha = .5, c = "cornflowerblue")
        axesRatio.scatter(rangeClusterNumberMetrics, ratioSingle,                        s = 4, label = "inter- over intra-cluster mean (single)", c = "navy")
        axesRatio.plot(   rangeClusterNumberMetrics, ratioSingle,                        ls = "--", alpha = .5, c = "navy")

        axesMeans.invert_xaxis()
        axesRatio.invert_xaxis()
        axesMeans.set_xlim(numberOfClustersHighest + 0.1, numberOfClustersLowest - 0.1)
        axesRatio.set_xlim(numberOfClustersHighest + 0.1, numberOfClustersLowest - 0.1)
        axesMeans.set_xticks(rangeClusterNumberMetrics)
        axesRatio.set_xticks(rangeClusterNumberMetrics)
        axesMeans.set_xticklabels([])
        axesMeans.set_ylabel("Gower's distance (1)")
        axesRatio.set_xlabel("number of clusters (1)")
        axesRatio.set_ylabel("ratio (1)")

        axesMeans.legend(loc = "upper left")
        axesRatio.legend(loc = "upper left")
        #axesMeans.set_title(r"distance measures during agglomerative hierarchical clustering $\vert$ " + r"\textbf{" + linkageType + r"}" + " linkage")
        axesMeans.set_title(r"inter- and intra-cluster distances $\vert$ " + r"\textbf{" + linkageType + r"}" + " linkage")
        axesMeans.grid(ls = "--", alpha = .2)
        axesRatio.grid(ls = "--", alpha = .2)

        #pyplot.subplots_adjust(left = 0.07, right = 0.98, bottom = 0.08, top = 0.92)
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "ProgressionMeanDistances" + figureExtension)
        pyplot.close()



    # Plot the clusters that the algorithm has found.
    if (plotClusters):
        for i in range(numberOfClustersLowest):
            figure           = pyplot.figure(1)

            figureWidth      = 4 + numberOfDimensions * .25 # in inch
            figureHeight     = 2 + clusterSizeList[i] * .1  # in inch

            numberOfRows     = int(numpy.round(figureHeight / figureHeightClustersBarCode))  # in 1
            numberOfColumns  = int(numpy.round(figureWidth  / figureWidthClustersColourBar)) # in 1
            print(numberOfRows, numberOfColumns)


            gridspec.GridSpec(numberOfRows, numberOfColumns)
            axesClusterWhole = pyplot.subplot2grid((numberOfRows, numberOfColumns), (0, 0), rowspan = numberOfRows - 1, colspan = numberOfColumns - 1)
            axesClusterMean  = pyplot.subplot2grid((numberOfRows, numberOfColumns), (numberOfRows - 1, 0), colspan = numberOfColumns - 1)
            axesColourBar    = pyplot.subplot2grid((numberOfRows, numberOfColumns), (0, numberOfColumns - 1), rowspan = numberOfRows - 1)

            axesClusterWhole.imshow(clusterMatrixPlotList[i], cmap = colourMapClusters, aspect = "auto", vmin = 0, vmax = 1, interpolation = "nearest")
            axesClusterMean.imshow(numpy.reshape(clusterMeansList[i], (1, numberOfDimensions)), cmap = colourMapClusters, aspect = "auto", vmin = 0, vmax = 1, interpolation = "nearest")

            colourBar        = colorbar.ColorbarBase(axesColourBar, cmap = colourMapClusters, orientation = "vertical")

            axesClusterWhole.set_xticks(numpy.arange(numberOfDimensions))
            axesClusterWhole.set_xticklabels([])
            axesClusterWhole.set_yticks(numpy.arange(clusterSizeList[i]))

            #if (clusterSizeList[i] > 100): # For dense clusters, observation labels are best omitted.
            #    axesClusterWhole.set_yticklabels([""] * clusterSizeList[i])
            #else: # For sparse clusters, observation labels are best included.
            axesClusterWhole.set_yticklabels(clusterObservationNamesList[i], fontsize = "xx-small")

            axesClusterMean.set_xticks(numpy.arange(numberOfDimensions))
            axesClusterMean.set_xticklabels(dimensionsUsed, rotation = 90)
            axesClusterMean.set_yticks([0])
            axesClusterMean.set_yticklabels(["cluster\nmean"])

            axesClusterWhole.set_ylim(-0.5, clusterSizeList[i] - 0.5)

            #axesClusterWhole.set_title(projectName + "\n" + r"Gower's distance agglomerative hierarchical clustering $\vert$ \textbf{" +
            #                           linkageType + r"} linkage $\vert$ cluster " + str(i + 1) + " of " + str(numberOfClustersLowest) + r" $\vert$ " + str(clusterSizeList[i]) + " observations")
            axesClusterWhole.set_title("cluster " + str(i + 1) + " of " + str(numberOfClustersLowest) + r" $\vert$ " + str(clusterSizeList[i]) + r" observations $\vert$ $\textbf{" + linkageType + r"}$ linkage")

            figure.set_size_inches(w = figureWidth, h = figureHeight)
            pyplot.tight_layout()
            #pyplot.subplots_adjust(left = 0.25, right = 0.97, bottom = 0.05, top = 0.93, hspace = 0.1)
            figure.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "Cluster" + str(i + 1) + figureExtension)
            pyplot.close()


            print ("Cluster " + str(i + 1) + " of " + str(numberOfClustersLowest) + " visualised!")
            print ("clusterSize:",             clusterSizeList[i])
            print ("clusterObservationNames:", clusterObservationNamesList[i])
            print ("clusterMeans:",            clusterMeansList[i])



    if (plotClusterMeans):
        figure            = pyplot.figure(figsize = (figureWidthClusterMeans, figureHeightClusterMeans))

        numberOfRows      = rowSpanGroups + rowSpanDimensions + numberOfClustersLowest * rowSpanCluster + rowSpanWhiteSpace + rowSpanColourBar
        numberOfColumns   = 3

        gridspec.GridSpec(numberOfRows, numberOfColumns)

        axesGroups        = pyplot.subplot2grid((numberOfRows, numberOfColumns), (0, 0), rowspan = rowSpanGroups, colspan = 3)

        groups            = numpy.zeros(numberOfDimensions)
        # Apply 'hacky' fix so that the last group name can be plotted.
        listGroupIndices  = listGroupIndices.copy()
        listGroupIndices.append(numberOfDimensions)
        # Plot group names.
        for i in range(len(listGroupIndices) - 1):
            groups[listGroupIndices[i] : ] += 1
            axesGroups.text((listGroupIndices[i] + listGroupIndices[i + 1] - 1) / 2, 0, listGroupNames[i], horizontalalignment = "center", verticalalignment = "center", fontsize = fontSizeGroups)

        axesGroups.imshow(numpy.reshape(groups, (1, numberOfDimensions)), cmap = colourMapClusterMeans, aspect = "auto", alpha = .75)
        axesGroups.set_xticks([])
        axesGroups.set_yticks([])

        axesDimensions = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups, 0), rowspan = rowSpanDimensions, colspan = 3)
        axesDimensions.imshow(numpy.reshape(groups, (1, numberOfDimensions)), cmap = colourMapClusterMeans, aspect = "auto", alpha = .5)
        axesDimensions.set_xticks([])
        axesDimensions.set_yticks([])

        for i in range(numberOfDimensions):
            axesDimensions.text(i, 0, dimensionsUsed[i], rotation = 90, horizontalalignment = "center", verticalalignment = "center", fontsize = fontSizeDimensions)

        for i in range(numberOfClustersLowest):
            axesCluster = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + i * rowSpanCluster, 0), rowspan = rowSpanCluster, colspan = 3)

            axesCluster.imshow(numpy.reshape(clusterMeansList[i], (1, numberOfDimensions)), cmap = colourMapClusters, aspect = "auto", vmin = 0, vmax = 1, alpha = 1)

            axesCluster.set_xticks([])
            axesCluster.set_yticks([0])
            axesCluster.set_yticklabels([r"\textbf{cluster\ " + str(i + 1) + r"}" + "\n" + r"$N = " + str(clusterSizeList[i]) + r"$"], fontsize = fontSizeClusters)
            axesCluster.tick_params(axis = "y", length = 0)

            if (plotClusterMeansText):
                for j, clusterMean in zip(range(numberOfDimensions), clusterMeansList[i]):
                    axesCluster.text(j, 0, numpy.round(clusterMean, 2), c = "gray", ha = "center", va = "center")


        axesWhiteSpace = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + numberOfClustersLowest * rowSpanCluster, 0), rowspan = rowSpanWhiteSpace, colspan = 3)
        axesWhiteSpace.set_visible(False)

        axesColourBar = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + numberOfClustersLowest * rowSpanCluster + rowSpanWhiteSpace, 2), rowspan = rowSpanColourBar, colspan = 1)
        axesColourBar.imshow(numpy.reshape(numpy.linspace(0, 1, num = 1000 + 1, endpoint = True), (1, 1000 + 1)), cmap = colourMapClusters, aspect = "auto", vmin = 0, vmax = 1, alpha = 1)
        axesColourBar.set_xticks([0, 1000])
        axesColourBar.set_xticklabels([0, 1])
        axesColourBar.set_yticks([])

        #figure.set_size_inches(w = 11.7 * 2, h = 8.3 * 2 * .95)
        pyplot.subplots_adjust(left = leftClusterMeans, right = rightClusterMeans, bottom = bottomClusterMeans, top = topClusterMeans)#, hspace = 0)
        figure.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "ClusterMeans" + figureExtension)

        pyplot.close()



    if (plotDistanceInterClustersMatrices):
        def plotDistanceInterClustersMatrix(distanceInterClustersMatrix, linkageTypeCurrent):
            '''
            Plots all distances between the clusters present at this iteration of the algorithm.
            '''

            distanceInterClustersMax = numpy.amax(distanceInterClustersMatrix[numpy.triu_indices(numberOfClustersLowest, k = 1)])

            pyplot.figure(figsize = (figureWidthDistanceInterClustersMatrix, figureHeightDistanceInterClustersMatrix))
            image                    = pyplot.imshow(distanceInterClustersMatrix / distanceInterClustersMax * 100, vmin = distanceVMin, vmax = 100, cmap = colourMapDistanceInterClustersMatrices)
            axesMain                 = pyplot.gca()
            axesColourBar            = make_axes_locatable(axesMain).append_axes("right", size = colourBarWidth, pad = colourBarDistance)
            colourBar                = pyplot.colorbar(image, cax = axesColourBar, label = r"distance relative to largest distance (\%)")

            for i in range(numberOfClustersLowest):
                for j in range(i + 1, numberOfClustersLowest):
                    axesMain.text(j, i, str(int(numpy.round(distanceInterClustersMatrix[i, j] / distanceInterClustersMax * 100))) + r"\%", c = "white", horizontalalignment = "center", verticalalignment = "center", fontsize = "x-small")

            axesMain.set_aspect("equal")
            axesMain.set_xlabel("cluster ID")
            axesMain.set_ylabel("cluster ID")
            axesMain.set_xticks(numpy.arange(numberOfClustersLowest))
            axesMain.set_yticks(numpy.arange(numberOfClustersLowest))
            axesMain.set_xticklabels(numpy.arange(numberOfClustersLowest) + 1)
            axesMain.set_yticklabels(numpy.arange(numberOfClustersLowest) + 1)
            axesMain.set_title(r"\textbf{" + linkageTypeCurrent + r" linkage distances between all pairs of " + str(numberOfClustersLowest) + " clusters}" + "\n" + r"Gower's distance metric $\vert$ " + linkageType + " linkage clustering")
            pyplot.subplots_adjust(left = 0.07, right = 0.91, bottom = 0.07, top = 0.93)
            pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "DistanceInterClustersMatrix" + linkageTypeCurrent.capitalize() + figureExtension)

            pyplot.close()

        plotDistanceInterClustersMatrix(distanceInterClustersMatrixComplete, "complete")
        plotDistanceInterClustersMatrix(distanceInterClustersMatrixAverage,  "average")
        plotDistanceInterClustersMatrix(distanceInterClustersMatrixSingle,   "single")


    if (plotDendrogram):
        # Plot clustering from 'numberOfClustersLowest' to 1 cluster.

        # Load dendrogram matrix.
        dendrogramMatrix           = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + "DendrogramMatrix.npy")

        # Select last few rows of dendrogram matrix.
        dendrogramMatrixPartialOld = dendrogramMatrix[-(numberOfClustersLowest - 1) : , : ]


        # Manipulate this last bit of the dendrogram matrix: rename cluster indices and change the inter-cluster distances of the third column.
        # See documentation of scipy.cluster.hierarchy.linkage.
        dendrogramMatrixPartialNew = numpy.copy(dendrogramMatrixPartialOld)

        # Change the 'large number' cluster indices so that they match the 'small number' cluster indices shown in other plots.
        # Without this change of indices, we can't trick 'scipy.cluster.hierarchy.linkage' into accepting this smaller dendrogram matrix.
        indicesOld                 = numpy.sort(dendrogramMatrixPartialOld[ : , : 2].flatten())
        indicesNew                 = numpy.arange(2 * (numberOfClustersLowest - 1))
        for indexOld, indexNew in zip(indicesOld, indicesNew):
            dendrogramMatrixPartialNew[dendrogramMatrixPartialOld == indexOld] = indexNew

        # Change distances to make the vertical steps in the dendrogram uniformly 1.
        dendrogramMatrixPartialNew[ : , 2] = numpy.arange(numberOfClustersLowest - 1) + 1


        # Plot dendrogram.
        pyplot.figure(figsize = (figureWidthDendrogram, figureHeightDendrogram))#(12, 8))#[6, 4, 5, 2, 3, 1, 0])
        hierarchy.dendrogram(dendrogramMatrixPartialNew, link_color_func = lambda k: colourDendrogram, labels = numpy.arange(numberOfClustersLowest) + 1)
        pyplot.xlabel("cluster ID")
        pyplot.ylabel("number of clusters (1)")

        # Set y-axis ticks and tick labels.
        yTicks                     = numpy.arange(numberOfClustersLowest)
        yTickLabels                = []
        for yTick in yTicks:
            yTickLabels.append(str(numberOfClustersLowest - yTick))
        pyplot.gca().set_yticks(yTicks)
        pyplot.gca().set_yticklabels(yTickLabels)

        pyplot.grid(ls = "--", axis = "y", alpha = .25)
        pyplot.title(r"dendrogram $\vert$ $\textbf{" + linkageType + "}$ linkage")
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClustersLowest) + "Dendrogram" + figureExtension)
        pyplot.close()
#numberOfRows                            = 6,
#numberOfColumns                         = 24,
#pyplot.ylabel("mean missing-data-corrected weighted Manhattan distance per weight (1)")
#pyplot.title(r"\textbf{" + projectName + "}" + "\n" + r"dendrogram after agglomerative hierarchical clustering $\vert$ " + linkageType + " linkage")
# We're differentiating between cluster indices and cluster IDs. A cluster's ID is always 1 higher than that cluster's index.
#IDNew = indexNew + 1
#indicesOld                 = numpy.sort(dendrogramMatrixPartialOld[ : , : 2].flatten())[ : numberOfClustersLowest]
#indicesNew                 = numpy.arange(numberOfClustersLowest)
#indicesNew[ : numberOfClustersLowest] = numpy.array([6, 4, 5, 2, 3, 1, 0])
#labels = ["cluster 7", "cluster 6", "cluster 4", "cluster 5", "cluster 2", "cluster 3", "cluster 1"], link_color_func = lambda k: "navy") #labels = organisationNames,labels = numpy.array([6, 4, 5, 2, 3, 1, 0]) + 1,
#numpy.linspace(1 / (numberOfClustersLowest - 1), 1, num = 6, endpoint = True)
'''
yTickLabels = (numpy.arange(numberOfClustersLowest) + 1)[::-1]

yTickLabelStrings = []
for yTickLabel in yTickLabels:
    yTickLabelStrings.append(str(yTickLabel))
yTickLabelStrings[0] = ""

pyplot.gca().set_yticks(yTickLabels)
pyplot.gca().set_yticklabels(yTickLabelStrings)#(numpy.arange(numberOfClustersLowest) + 1)[::-1])
'''
'''
print(dendrogramMatrix.shape)
print(indicesOld)
print(indicesNew)
print(dendrogramMatrixPartialOld)
print(dendrogramMatrixPartialNew)
print(dendrogramMatrixPartialOld == 1)
'''
#projectName                             = "",          # name of the AHC project




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

Calculates and visualises the non-randomness of the agglomerative hierarchical clustering by means of a resampling simulation.

clusterList:  numpy.ndarray of lists, contains cluster observation indices
clusterSizes: numpy.ndarray of ints,  contains cluster sizes
'''

from matplotlib import pyplot
import matplotlib, numpy, os
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["savefig.dpi"] = 400

def AHCResampling(directoryData,
                  directoryFigures,
                  linkageType,
                  numberOfClusters,
                  dimensionsUsed,
                  listGroupNames,                     # names   of the dimension groups
                  listGroupIndices,                   # indices of the dimension groups
                  dataSetName             = "full",
                  numberOfResamples       = 1000,     # number of random clusterings to simulate
                  numberOfObservationsMin = 15,       # number of observations required for a cluster on a dimension to show the significance
                  numberOfSigmataMax      = 9,        # number of standard deviations that should correspond with the extremes of the colour map
                  numberOfSigmataTick     = 3,        # number of standard deviations that should correspond with an extra positive and negative tick
                  rowSpanGroups           = 2,        # number of rows occupied by the names of grouped dimensions
                  rowSpanDimensions       = 10,       # number of rows occupied by the names of dimensions
                  rowSpanCluster          = 5,        # number of rows occupied by a cluster
                  rowSpanWhiteSpace       = 1,        # number of rows occupied by the whitespace between the clusters and the colour bar
                  rowSpanColourBar        = 1,        # number of rows occupied by the colour bar
                  numberOfColumns         = 3,        # number of columns for the whole plot; used to control the colour bar length
                  plotScattersExtension   = ".pdf",   # file format of scatter plots (".pdf" or ".png")
                  plotBarcodesExtension   = ".pdf",   # file format of barcode plots (".pdf" or ".png")
                  fontSizeGroups          = 12,
                  fontSizeDimensions      = 12,
                  fontSizeClusters        = 12,
                  plotScatters            = True,
                  plotBarcodes            = True,
                  figureWidthScatters     = 6,        # in inch
                  figureHeightScatters    = 4,        # in inch
                  figureWidthBarcodes     = 10,       # in inch
                  figureHeightBarcodes    = 8,        # in inch
                  colourMapBarcodes       = "Greens", # colour map for background of dimension names and dimension group names
                  colourMapSignificances  = "RdBu_r", # colour map for significances
                  leftBarcodes            = .08,      # in 1
                  rightBarcodes           = .985,     # in 1
                  bottomBarcodes          = .04,      # in 1
                  topBarcodes             = .99,      # in 1
                  plotBarcodesText        = True
                  ):
    '''
    '''

    linkageTypeCapitalised     = linkageType.capitalize()

    # Load data.
    # Load 'observationMatrix'.
    observationMatrix          = numpy.load(directoryData + dataSetName + "/" + "observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape

    nameFileOutput             = "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList" + ".npy"
    clusterList                = numpy.load(directoryData + dataSetName + "/" + nameFileOutput, allow_pickle = True)
    clusterSizes               = numpy.empty(numberOfClusters, dtype = int) # in 1
    for i in range(numberOfClusters):
        clusterSizes[i] = len(clusterList[i])


    # Resample.
    clusterResampledMeans      = []
    for i in range(numberOfClusters):
        clusterResampledMeans.append(numpy.empty((numberOfResamples, numberOfDimensions)))

    for i in range(numberOfResamples):
        observationMatrixShuffled = numpy.random.permutation(observationMatrix)

        for j in range(numberOfClusters):
            indexStart = numpy.sum(clusterSizes[0 : j])
            indexEnd   = numpy.sum(clusterSizes[0 : j + 1])

            clusterResampledMeans[j][i, : ] = numpy.nanmean(observationMatrixShuffled[indexStart : indexEnd], axis = 0)


    # Characterise mean and standard deviation of resample means.
    clusterResampledMeansMeans = numpy.empty((numberOfClusters, numberOfDimensions))
    clusterResampledMeansSDs   = numpy.empty((numberOfClusters, numberOfDimensions))

    for i in range(numberOfClusters):
        clusterResampledMeansMeans[i, : ] = numpy.nanmean(clusterResampledMeans[i], axis = 0)
        clusterResampledMeansSDs[i, : ]   = numpy.nanstd(clusterResampledMeans[i], axis = 0)



    # Generate the directory for the figures, if it does not exist yet.
    if (not os.path.exists(directoryFigures + dataSetName + "/")):
        os.makedirs(directoryFigures + dataSetName + "/")


    if (plotScatters):
        for i in range(numberOfClusters):
            pyplot.figure(figsize = (figureWidthScatters, figureHeightScatters))

            pyplot.scatter(range(numberOfDimensions), numpy.nanmean(observationMatrix[clusterList[i]], axis = 0), c = "mediumseagreen", zorder = 2)
            for j in range(numberOfResamples):
                pyplot.scatter(range(numberOfDimensions), clusterResampledMeans[i][j], c = "gray", alpha = .05, zorder = 1)

            pyplot.scatter(range(numberOfDimensions), clusterResampledMeansMeans[i], facecolors = "none", edgecolors = "black")
            pyplot.xlim(-.5, numberOfDimensions - .5)
            pyplot.ylim(-0.05, 1.05)

            pyplot.gca().set_xticks(range(numberOfDimensions))
            pyplot.gca().set_xticklabels(dimensionsUsed, rotation = 90, fontsize = 6)

            pyplot.title("cluster " + str(i + 1) + " of " + str(numberOfClusters) + r" $\vert$ " + str(clusterSizes[i]) + r" observations $\vert$ $\textbf{" + linkageType + "}$ linkage")
            pyplot.grid(ls = "--", alpha = .2)
            pyplot.tight_layout()
            pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterSignificance" + str(i + 1) + plotScattersExtension)
            pyplot.close()


    if (plotBarcodes):
        figure           = pyplot.figure(figsize = (figureWidthBarcodes, figureHeightBarcodes))

        numberOfRows     = rowSpanGroups + rowSpanDimensions + numberOfClusters * rowSpanCluster + rowSpanWhiteSpace + rowSpanColourBar

        axesGroups       = pyplot.subplot2grid((numberOfRows, numberOfColumns), (0, 0), rowspan = rowSpanGroups, colspan = 3)
        groups           = numpy.zeros(numberOfDimensions)
        # Apply 'hacky' fix so that the last group name can be plotted.
        listGroupIndices = listGroupIndices.copy()
        listGroupIndices.append(numberOfDimensions)
        # Plot group names.
        for i in range(len(listGroupIndices) - 1):
            groups[listGroupIndices[i] : ] += 1
            axesGroups.text((listGroupIndices[i] + listGroupIndices[i + 1] - 1) / 2, 0, listGroupNames[i], horizontalalignment = "center", verticalalignment = "center", fontsize = fontSizeGroups)

        axesGroups.imshow(numpy.reshape(groups, (1, numberOfDimensions)), cmap = colourMapBarcodes, aspect = "auto", alpha = .75)
        axesGroups.set_xticks([])
        axesGroups.set_yticks([])

        axesDimensions  = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups, 0), rowspan = rowSpanDimensions, colspan = 3)
        axesDimensions.imshow(numpy.reshape(groups, (1, numberOfDimensions)), cmap = colourMapBarcodes, aspect = "auto", alpha = .5)
        axesDimensions.set_xticks([])
        axesDimensions.set_yticks([])

        for i in range(numberOfDimensions):
            axesDimensions.text(i, 0, dimensionsUsed[i], rotation = 90, horizontalalignment = "center", verticalalignment = "center", fontsize = fontSizeDimensions)

        for clusterIndex in range(numberOfClusters):
            axesCluster           = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + clusterIndex * rowSpanCluster, 0), rowspan = rowSpanCluster, colspan = 3)

            # Calculate, for this cluster and for each dimension, the (masked) significance of the deviation of the observed mean from the mean of the resampled means.
            numbersOfObservations = clusterSizes[clusterIndex] - numpy.sum(numpy.isnan(observationMatrix[clusterList[clusterIndex]]), axis = 0) # for this cluster, the number of observations available for each dimension; not to be confused with 'numberOfObservations'
            significances         = (numpy.nanmean(observationMatrix[clusterList[clusterIndex]], axis = 0) - clusterResampledMeansMeans[clusterIndex]) / clusterResampledMeansSDs[clusterIndex]
            significances[numbersOfObservations < numberOfObservationsMin] = numpy.nan
            significances         = numpy.reshape(significances, (1, numberOfDimensions))

            # Plot, for this cluster, bar code with significances.
            axesCluster.imshow(significances, cmap = colourMapSignificances, aspect = "auto", vmin = -1 * numberOfSigmataMax, vmax = numberOfSigmataMax, alpha = 1)

            axesCluster.set_xticks([])
            axesCluster.set_yticks([0])
            axesCluster.set_yticklabels([r"\textbf{cluster\ " + str(clusterIndex + 1) + r"}" + "\n" + r"$N = " + str(clusterSizes[clusterIndex]) + r"$"], fontsize = fontSizeClusters)
            axesCluster.tick_params(axis = "y", length = 0)

            if (plotBarcodesText):
                for j, significance in zip(range(numberOfDimensions), significances[0]):
                    axesCluster.text(j, 0, numpy.round(significance, 1), c = "gray", ha = "center", va = "center")


        axesWhiteSpace  = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + numberOfClusters * rowSpanCluster, 0), rowspan = rowSpanWhiteSpace, colspan = 3)
        axesWhiteSpace.set_visible(False)

        axesColourBar   = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + numberOfClusters * rowSpanCluster + rowSpanWhiteSpace, 2), rowspan = rowSpanColourBar, colspan = 1)
        numberOfColours = 1000 + 1 # in 1
        axesColourBar.imshow(numpy.reshape(numpy.linspace(-1 * numberOfSigmataMax, numberOfSigmataMax, num = numberOfColours, endpoint = True), (1, numberOfColours)), cmap = colourMapSignificances, aspect = "auto", vmin = -1 * numberOfSigmataMax, vmax = numberOfSigmataMax, alpha = 1)
        axesColourBar.set_xticks([-.5, (numberOfColours - 1) / 2 - numberOfColours / (2 * numberOfSigmataMax) * numberOfSigmataTick, (numberOfColours - 1) / 2, (numberOfColours - 1) / 2 + numberOfColours / (2 * numberOfSigmataMax) * numberOfSigmataTick, numberOfColours - .5])
        axesColourBar.set_xticklabels([r"$-" + str(numberOfSigmataMax) + r"\ \sigma$", r"$-" + str(numberOfSigmataTick) + "\ \sigma$", r"$0$", r"$" + str(numberOfSigmataTick) + "\ \sigma$", r"$" + str(numberOfSigmataMax) + r"\ \sigma$"])
        axesColourBar.set_yticks([])

        #figure.set_size_inches(w = 11.7 * 2, h = 8.3 * 2 * .95)
        pyplot.subplots_adjust(left = leftBarcodes, right = rightBarcodes, bottom = bottomBarcodes, top = topBarcodes)

        figure.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterSignificances" + plotBarcodesExtension)

        pyplot.close()




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.
'''

from matplotlib import cm, colorbar, pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy, numpy, os
import matplotlib
matplotlib.rcParams["text.usetex"] = True


def AHCDataExploration(directoryData,
                       directoryFigures,
                       dataSetName,
                       dimensionsUsed,
                       plotObservationMatrix  = True,
                       plotAvailabilityMatrix = True,
                       projectName            = "",
                       colourBarWidth         = "2.5%",
                       colourBarDistance      = .06,
                       colourMap              = cm.coolwarm,
                       colourMapBinary        = cm.RdYlGn):
    '''
    '''

    colourMap = copy.copy(cm.get_cmap(colourMap))
    colourMap.set_bad(color = "white")

    # Load 'observationMatrix' and 'hasObservation'.
    observationMatrix = numpy.load(directoryData + dataSetName + "/" + "observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape
    hasObservation    = loadAvailabilityMatrix(observationMatrix)
    observationNames  = numpy.arange(numberOfObservations)


    # Generate the directory for the figures, if it does not exist yet.
    if (not os.path.exists(directoryFigures + dataSetName + "/")):
        os.makedirs(directoryFigures + dataSetName + "/")


    # Plots the observation matrix.
    if (plotObservationMatrix):
        # As not all observation matrices have 'numpy.float64' entries, we create a copy of the observation matrix for which this is true.
        observationMatrixPlot = observationMatrix.astype(numpy.float64)


        pyplot.figure(figsize = (numberOfDimensions * .3, numberOfObservations * .3))
        pyplot.imshow(observationMatrixPlot, cmap = colourMap, aspect = "auto")
        axesMain              = pyplot.gca()
        axesColourBar         = make_axes_locatable(axesMain).append_axes("right", size = colourBarWidth, pad = colourBarDistance)
        '''
        boundaries            = [0, 1, 2, 3]
        colours               = colourMap(numpy.linspace(0, 1, endpoint = True, num = 4))
        colourBarMap          = colors.ListedColormap(colours)
        colourBar             = colorbar.ColorbarBase(axesColourBar, cmap = colourBarMap, orientation = "vertical", ticks = [0.125, 0.375, 0.625, 0.875])
        colourBar.set_ticklabels(boundaries)
        '''
        colourBar             = colorbar.ColorbarBase(axesColourBar, cmap = colourMap, orientation = "vertical")

        axesMain.set_xticks(numpy.arange(numberOfDimensions))
        axesMain.set_xticklabels(dimensionsUsed, rotation = 90)
        axesMain.set_yticks(numpy.arange(numberOfObservations))
        axesMain.set_yticklabels(observationNames)
        axesMain.set_ylim(-0.5, numberOfObservations - 0.5) # use margins of 0.5 to ensure the first and last row are shown fully
        axesMain.invert_yaxis()
        axesMain.set_title(projectName + "\nobservation matrix")
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + "ObservationMatrix" + ".pdf")
        pyplot.close()


    # Plots the availability matrix.
    if (plotAvailabilityMatrix):
        pyplot.figure(figsize = (numberOfDimensions * .3, numberOfObservations * .3))
        pyplot.imshow(hasObservation, cmap = colourMapBinary, aspect = "auto")
        axesMain      = pyplot.gca()
        axesColourBar = make_axes_locatable(axesMain).append_axes("right", size = colourBarWidth, pad = colourBarDistance)
        colourBar     = colorbar.ColorbarBase(axesColourBar, cmap = colourMapBinary, orientation = "vertical")

        axesMain.set_xticks(numpy.arange(numberOfDimensions))
        axesMain.set_xticklabels(dimensionsUsed, rotation = 90)
        axesMain.set_yticks(numpy.arange(numberOfObservations))
        axesMain.set_yticklabels(observationNames)
        axesMain.set_ylim(-0.5, numberOfObservations - 0.5) # use margins of 0.5 to ensure the first and last row are shown fully
        axesMain.invert_yaxis()
        axesMain.set_title(projectName + "\nobservation availability matrix")
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + "ObservationAvailabilityMatrix" + ".pdf")
        pyplot.close()




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.
'''

from matplotlib import cm, pyplot
import matplotlib, numpy, os, pandas
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["savefig.dpi"] = 400

def AHCResultsRawData(directoryData,
                      directoryFigures,
                      fileName,
                      linkageType,
                      numberOfClusters,
                      indexColumnStart            = None,
                      indexColumnEnd              = None,
                      dataSetName                 = "full",
                      numberOfDimensionsPerFigure = 9,
                      colourMap                   = cm.Spectral,
                      faceColour                  = "0.8",
                      numberOfBins                = 10,
                      colourBinText               = "white",
                      figureExtension             = ".pdf"):
    '''
    '''

    linkageTypeCapitalised      = linkageType.capitalize()

    # Load data.
    clusterList                 = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList" + ".npy", allow_pickle = True)
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)


    # Generate the directory for the figures, if it does not exist yet.
    if (not os.path.exists(directoryFigures + dataSetName + "/")):
        os.makedirs(directoryFigures + dataSetName + "/")

    # Create figures.
    colours                     = colourMap(numpy.linspace(0, 1, num = numberOfClusters, endpoint = True))
    numberOfFigures             = int(numpy.ceil(numberOfDimensions / numberOfDimensionsPerFigure))

    for indexFigure in range(numberOfFigures):
        pyplot.figure(figsize = (3 * numberOfDimensionsPerFigure, 2 * numberOfClusters))

        for indexDimension in range(indexFigure * numberOfDimensionsPerFigure, min((indexFigure + 1) * numberOfDimensionsPerFigure, numberOfDimensions)):

            observationsDimension = observationMatrix[ : , indexDimension]
            binMinEdgeLeft        = numpy.nanmin(observationsDimension)
            binMaxEdgeRight       = numpy.nanmax(observationsDimension)
            bins                  = numpy.linspace(binMinEdgeLeft, binMaxEdgeRight, num = numberOfBins + 1, endpoint = True)


            for indexCluster in range(numberOfClusters):
                axes                     = pyplot.subplot2grid((numberOfClusters, numberOfDimensionsPerFigure), (indexCluster, indexDimension - indexFigure * numberOfDimensionsPerFigure), rowspan = 1, colspan = 1)
                observationsPlot         = observationMatrix[clusterList[indexCluster], indexDimension]
                numberOfObservationsPlot = len(observationsPlot) # in 1
                numberOfNaNsPlot         = numpy.sum(numpy.isnan(observationsPlot))
                if (numberOfNaNsPlot < numberOfObservationsPlot):
                    n, bins, patches = axes.hist(observationMatrix[clusterList[indexCluster], indexDimension], color = colours[indexCluster], bins = bins)
                    for indexBin in range(numberOfBins):
                        if (int(n[indexBin]) > 0):
                            axes.text((bins[indexBin] + bins[indexBin + 1]) / 2, .02 * axes.get_ylim()[1], int(n[indexBin]), horizontalalignment = "center", verticalalignment = "bottom", rotation = 90, c = colourBinText, fontsize = "xx-small")

                axes.set_facecolor(faceColour)

                stringTitle = r"$N$: " + str(numberOfObservationsPlot - numberOfNaNsPlot) + r" $\vert$ " + r"$\mu$: " + str(numpy.round(numpy.nanmean(observationsPlot), 2)) + r" $\vert$ " + r"$\sigma$: " + str(numpy.round(numpy.nanstd(observationsPlot), 2))
                if (indexCluster == 0):
                    stringTitle = r"\textbf{" + dimensionsUsed[indexDimension] + r"}" + "\n" + stringTitle
                axes.set_title(stringTitle)

                if (indexDimension - indexFigure * numberOfDimensionsPerFigure == 0):
                    axes.set_ylabel(r"\textbf{cluster} " + str(indexCluster + 1))

        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClustersRawData" + str(indexFigure) + figureExtension)
        pyplot.close()




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

Prepare uniform data sets.
'''

import numpy, os

def AHCPrepareDataSetUniform(directoryData,
                             fileName,
                             numberOfDataSets,
                             indexColumnStart = None,
                             indexColumnEnd   = None,
                             numberOfNumerals = 3
                             ):
    '''
    '''

    # Load data set.
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)


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
        dataSetName              = "uniform" + "_" + str(indexDataSet).zfill(numberOfNumerals)
        directoryDataSet         = directoryData + dataSetName + "/"
        if (not os.path.exists(directoryDataSet)):
            os.makedirs(directoryDataSet)
        numpy.save(directoryDataSet + "observationMatrix.npy", observationMatrixUniform)

        # Report progress.
        print("Saved data set " + dataSetName + " in " + directoryDataSet + "!")




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

This program compares the coefficients of determination obtained with clustering of uniform random data to clustering of actual data.
'''

from matplotlib import pyplot
import matplotlib, numpy
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["savefig.dpi"] = 400

def AHCResultsTestUniformity(directoryData,
                             directoryFigures,
                             linkageType,
                             numberOfDataSets,
                             numberOfNumerals,
                             numberOfClustersHighest,
                             numberOfClustersLowest,
                             colourActual    = "mediumseagreen",
                             colourUniform   = "crimson",
                             alphaActual     = .5,
                             alphaUniform    = .2,
                             figureWidth     = 6, # in inch
                             figureHeight    = 3, # in inch
                             figureExtension = ".pdf"
                             ):
    '''
    '''
    linkageTypeCapitalised             = linkageType.capitalize()
    numberOfData                       = numberOfClustersHighest - numberOfClustersLowest + 1 # in 1
    coefficientsOfDeterminationActual  = numpy.empty(numberOfData) # in 1
    coefficientsOfDeterminationUniform = numpy.empty((numberOfDataSets, numberOfData)) # in 1
    rangeClusterNumberMetrics          = numpy.arange(numberOfClustersLowest, numberOfClustersHighest + 1)[ : : -1]


    # Load data set names.
    dataSetNames                       = []
    for indexDataSet in range(numberOfDataSets):
        dataSetNames.append("uniform" + "_" + str(indexDataSet).zfill(numberOfNumerals))


    # Load coefficients of determination.
    for numberOfClusters, j in zip(rangeClusterNumberMetrics, range(numberOfData)):
        metricsSingleNumber                     = numpy.load(directoryData + "full" + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "MetricsSingleNumber" + ".npy")
        coefficientsOfDeterminationActual[j] = metricsSingleNumber[13]

    for dataSetName, i in zip(dataSetNames, range(numberOfDataSets)):
        for numberOfClusters, j in zip(rangeClusterNumberMetrics, range(numberOfData)):
            metricsSingleNumber                      = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "MetricsSingleNumber" + ".npy")
            coefficientsOfDeterminationUniform[i, j] = metricsSingleNumber[13]


    # Print statistics.
    print(numpy.mean(coefficientsOfDeterminationUniform, axis = 0))
    print(numpy.mean(coefficientsOfDeterminationUniform, axis = 0) - 2 * numpy.std(coefficientsOfDeterminationUniform, axis = 0))
    print(numpy.mean(coefficientsOfDeterminationUniform, axis = 0) + 2 * numpy.std(coefficientsOfDeterminationUniform, axis = 0))


    # Plot figure.
    pyplot.figure(figsize = (figureWidth, figureHeight))

    pyplot.scatter(rangeClusterNumberMetrics, coefficientsOfDeterminationActual * 100, s = 4, c = colourActual, label = "actual data")
    pyplot.plot(rangeClusterNumberMetrics, coefficientsOfDeterminationActual * 100, ls = "-", c = colourActual, alpha = alphaActual)

    for i in range(numberOfDataSets):
        if (i == 0):
            pyplot.scatter(rangeClusterNumberMetrics, coefficientsOfDeterminationUniform[i] * 100, s = 4, c = colourUniform, label = "uniform data\n(" + str(numberOfDataSets) + " realisations)")
        else:
            pyplot.scatter(rangeClusterNumberMetrics, coefficientsOfDeterminationUniform[i] * 100, s = 4, c = colourUniform)
        pyplot.plot(rangeClusterNumberMetrics, coefficientsOfDeterminationUniform[i] * 100, ls = "-", c = colourUniform, alpha = alphaUniform)

    pyplot.gca().invert_xaxis()
    pyplot.grid(ls = "--", alpha = .2)
    pyplot.xticks(rangeClusterNumberMetrics)#, fontsize = "x-large")
    #pyplot.yticks(fontsize = "x-large")
    pyplot.xlim(rangeClusterNumberMetrics[0] + .1, rangeClusterNumberMetrics[-1] - .1)
    #pyplot.ylim(0, 50)
    pyplot.xlabel("number of clusters (1)")#, fontsize = "x-large")
    pyplot.ylabel(r"fraction of explained variance (\%)")#, fontsize = "x-large")
    pyplot.legend(loc = "upper right", borderpad = .15)#, fontsize = "x-large")
    pyplot.title(r"goodness of fit: coefficient of determination $R^2$ $\vert$ $\textbf{" + linkageType + "}$ linkage")
    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageTypeCapitalised + "CoefficientsOfDeterminationSignificance" + figureExtension)
    pyplot.close()




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

Prepare jackknife data sets.
'''

import numpy, os

def AHCPrepareDataSetJackknife(directoryData,
                               fileName,
                               numberOfDataSets,
                               numberOfObservationsSubset,
                               indexColumnStart = None,
                               indexColumnEnd   = None,
                               numberOfNumerals = 3
                               ):
    '''
    '''

    # Load full data set.
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)

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




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.
'''

import numpy
from scipy import stats
from matplotlib import cm, pyplot
import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["savefig.dpi"] = 400


def AHCResultsJackknife(directoryData,
                        directoryFigures,
                        linkageType,
                        numberOfClusters,
                        numberOfDataSets,
                        numberOfObservationsSubset,
                        numberOfNumerals     = 3,
                        numberOfDecimals     = 1,
                        probabilityMixingMin = 0,
                        probabilityMixingMax = .3,
                        colourMapMixing      = "Reds",
                        colourTextMixing     = ".4",
                        colourBarWidth       = "2.5%",
                        colourBarDistance    = 0.06,
                        showRanks            = False,
                        figureWidthMixing    = 6,  # in inch
                        figureHeightMixing   = 6,  # in inch
                        figureWidthAll       = 12, # in inch
                        figureHeightAll      = 9,  # in inch
                        figureExtension      = ".pdf"
                        ):
    '''
    '''

    # Load the cluster list of the full data set.
    clusterListFull                = numpy.load(directoryData + "full/AHC" + linkageType.capitalize() + str(numberOfClusters) + "ClusterList.npy", allow_pickle = True)


    # Find the integer 'numberOfObservations', representing the number of observations in the full data set.
    numberOfObservations           = 0
    for indexCluster in range(numberOfClusters):
        indicesObservations   = clusterListFull[indexCluster]
        numberOfObservations += len(indicesObservations)

    # Generate the 1D array 'indicesCluster', which is an alternative way to store the same information present in 'clusterListFull'.
    indicesCluster                 = numpy.empty(numberOfObservations, dtype = numpy.uint16)
    for indexCluster in range(numberOfClusters):
        indicesObservations                 = clusterListFull[indexCluster]
        indicesCluster[indicesObservations] = indexCluster

    # Generate the boolean matrix 'inSameCluster' that stores, for each pair of observations, whether they are clustered together in the full data set.
    inSameCluster                  = (indicesCluster[ : , None] == indicesCluster[None, : ])


    # Count, for each observation pair, how often the pair occurs in the pool of jackknife subsets.
    inSameJackknifeSubsetCount     = numpy.zeros((numberOfObservations, numberOfObservations), dtype = numpy.uint32)
    # Count, for each observation pair, how often the pair is put in the same cluster (when it appears in the jackknife subsets).
    inSameJackknifeClusterCount    = numpy.zeros((numberOfObservations, numberOfObservations), dtype = numpy.uint32)

    # Iterate over clustering results of jackknife subsets.
    for indexDataSet in range(numberOfDataSets):
        directoryDataSet            = directoryData + "jackknife" + str(numberOfObservationsSubset) + "_" + str(indexDataSet).zfill(numberOfNumerals) + "/"

        # Increase 'inSameJackknifeSubsetCount' by 1 for each observation pair that occurs in the current jackknife subset.
        indicesObservations         = numpy.load(directoryDataSet + "indicesObservations.npy")
        indicesObservationsTiled    = numpy.tile(indicesObservations, numberOfObservationsSubset)
        indicesObservationsRepeated = numpy.repeat(indicesObservations, numberOfObservationsSubset)
        inSameJackknifeSubsetCount[indicesObservationsTiled, indicesObservationsRepeated] += 1

        # Load the clustering result of the current jackknife subset.
        clusterList                 = numpy.load(directoryDataSet + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "ClusterList.npy", allow_pickle = True)
        # Iterate over current jackknife subset clusters.
        for indexCluster in range(numberOfClusters):
            # Find the full data set indices of the observations in this cluster.
            indicesObservationsCluster         = indicesObservations[clusterList[indexCluster]]

            numberOfObservationsCluster        = len(indicesObservationsCluster)

            # Increase 'inSameJackknifeClusterCount' by 1 for each observation pair that occurs in the current cluster.
            indicesObservationsClusterTiled    = numpy.tile(indicesObservationsCluster, numberOfObservationsCluster)
            indicesObservationsClusterRepeated = numpy.repeat(indicesObservationsCluster, numberOfObservationsCluster)
            inSameJackknifeClusterCount[indicesObservationsClusterTiled, indicesObservationsClusterRepeated] += 1


    # Calculate, for each observation pair, what fraction of jackknife subsets that contain the pair, has the pair clustered together.
    inSameJackknifeClusterFraction = inSameJackknifeClusterCount / numpy.maximum(inSameJackknifeSubsetCount, 1)


    # Calculate, for each observation, what fraction of its relations with other (!) observations have been correct (i.e. as they were in the full data set clustering).
    correctRelationFraction        = (numpy.sum(inSameCluster * inSameJackknifeClusterCount + numpy.logical_not(inSameCluster) * (inSameJackknifeSubsetCount - inSameJackknifeClusterCount), axis = 0) - numpy.diag(inSameJackknifeSubsetCount)) / (numpy.sum(inSameJackknifeSubsetCount, axis = 0) - numpy.diag(inSameJackknifeSubsetCount))


    # Calculate, for each full data set cluster pair, what fraction of all pairs of their constituents in the jackknife subsets were ('wrongly') clustered together. We call this 'mixing'.
    mixingMatrix                   = numpy.zeros((numberOfClusters, numberOfClusters))
    mixingMatrix[numpy.tril_indices(numberOfClusters, k = 0)] = numpy.nan
    for i in range(numberOfClusters):
        indicesObservations1 = clusterListFull[i]
        for j in range(i + 1, numberOfClusters):
            indicesObservations2                = clusterListFull[j]
            inSameJackknifeClusterCountClusters = inSameJackknifeClusterCount[indicesObservations1][ : , indicesObservations2]
            inSameJackknifeSubsetCountClusters  = inSameJackknifeSubsetCount[indicesObservations1][ : , indicesObservations2]
            mixingMatrix[i, j]                  = numpy.sum(inSameJackknifeClusterCountClusters) / numpy.sum(inSameJackknifeSubsetCountClusters)
    mixingMatrixRank               = (numberOfClusters * (numberOfClusters - 1) / 2 - stats.rankdata(mixingMatrix).reshape((numberOfClusters, numberOfClusters)) + 1).astype(int)



    # Visualise results.
    pyplot.figure(figsize = (figureWidthMixing, figureHeightMixing))
    image                          = pyplot.imshow(mixingMatrix * 100, cmap = colourMapMixing, vmin = probabilityMixingMin * 100, vmax = probabilityMixingMax * 100)
    axesMain                       = pyplot.gca()
    axesColourBar                  = make_axes_locatable(axesMain).append_axes("right", size = colourBarWidth, pad = colourBarDistance)
    colourBar                      = pyplot.colorbar(image, cax = axesColourBar, label = r"mixing probability (\%)")

    #objectColourBar = pyplot.colorbar()#, size = "x-large")#, labelsize = "x-large")
    #objectColourBar.ax.tick_params(labelsize = 22)#"xx-large")
    #objectColourBar.set_label(label = "mixing probability (1)")#, size = 22)#"xx-large")

    for i in range(numberOfClusters):
        for j in range(i + 1, numberOfClusters):
            if (showRanks):
                axesMain.text(j, i, r"\textbf{rank " + str(mixingMatrixRank[i, j]) + ":}\n" + str(numpy.round(mixingMatrix[i, j] * 100, numberOfDecimals)) + r"\%", c = colourTextMixing, horizontalalignment = "center", verticalalignment = "center", fontsize = "x-small")
            else:
                axesMain.text(j, i, str(numpy.round(mixingMatrix[i, j] * 100, numberOfDecimals)) + r"\%", c = colourTextMixing, horizontalalignment = "center", verticalalignment = "center", fontsize = "x-small")
    #pyplot.xlabel("full data set cluster ID")
    #pyplot.ylabel("full data set cluster ID")
    axesMain.set_xlabel("cluster ID")#, fontsize = 22)#"xx-large")
    axesMain.set_ylabel("cluster ID")#, fontsize = 22)#"xx-large")
    axesMain.set_xticks(numpy.arange(numberOfClusters))
    axesMain.set_xticklabels(numpy.arange(numberOfClusters) + 1)
    axesMain.set_yticks(numpy.arange(numberOfClusters))
    axesMain.set_yticklabels(numpy.arange(numberOfClusters) + 1)
    #pyplot.xticks(fontsize = 22)#"xx-large")
    #pyplot.yticks(fontsize = 22)#"xx-large")
    axesMain.set_title(r"\textbf{jackknife analysis}" + "\n" + r"\textnumero\ observations in subset: " + str(numberOfObservationsSubset) + r" (" + str(numpy.round(numberOfObservationsSubset / numberOfObservations * 100, 1)) + "\% of " + str(numberOfObservations) + r") $\vert$ " + linkageType + r" linkage")#$\vert$ \textnumero\ observations in total: " + str(numberOfObservations) + r"
    #pyplot.title(r"\textbf{agglomerative hierarchical clustering} jackknife analysis" + "\n" + linkageType + r" linkage $\vert$ \textnumero\ clusters: " + str(numberOfClusters) + r" $\vert$ \textnumero\ observations in subset: " + str(numberOfObservationsSubset))
    #pyplot.tight_layout()
    pyplot.subplots_adjust(left = .07, right = .91, bottom = .07, top = .93)
    pyplot.savefig(directoryFigures + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "Jackknife" + str(numberOfObservationsSubset) + "Mixing" + figureExtension)
    pyplot.close()


    pyplot.figure(figsize = (figureWidthAll, figureHeightAll))
    pyplot.imshow(inSameJackknifeClusterFraction, cmap = "cividis", vmin = 0, vmax = 1)
    pyplot.colorbar(label = "probability of staying together in jackknife clusterings (1)")
    pyplot.xlabel("full data set observation index (1)")
    pyplot.ylabel("full data set observation index (1)")
    pyplot.title(r"\textbf{agglomerative hierarchical clustering} jackknife analysis" + "\n" + linkageType + r" linkage $\vert$ \textnumero\ clusters: " + str(numberOfClusters) + r" $\vert$ \textnumero\ observations in subset: " + str(numberOfObservationsSubset))
    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "Jackknife" + str(numberOfObservationsSubset) + "All" + figureExtension)
    pyplot.close()


    figureObject, axesGrid         = pyplot.subplots(numberOfClusters, 3, figsize = (12, numberOfClusters * 4))
    binEdges1                      = numpy.linspace(0, 1, num = 10 + 1, endpoint = True) # in 1
    binEdges2                      = numpy.linspace(.5, 1, num = 20 + 1, endpoint = True) # in 1

    # Iterate over full data set clusters.
    for indexCluster in range(numberOfClusters):
        indicesObservations                         = clusterListFull[indexCluster]
        numberOfObservationsCluster                 = len(indicesObservations)
        indicesObservationsTiled                    = numpy.tile(indicesObservations, numberOfObservationsCluster)
        indicesObservationsRepeated                 = numpy.repeat(indicesObservations, numberOfObservationsCluster)

        inSameJackknifeClusterFractionCluster       = inSameJackknifeClusterFraction[indicesObservationsTiled, indicesObservationsRepeated].reshape((numberOfObservationsCluster, numberOfObservationsCluster))
        inSameJackknifeClusterFractionClusterUnique = inSameJackknifeClusterFractionCluster[numpy.triu_indices(numberOfObservationsCluster, k = 1)]
        binCounts1, binEdges1                       = numpy.histogram(inSameJackknifeClusterFractionClusterUnique, bins = binEdges1)

        axesGrid[indexCluster, 0].bar((binEdges1[1 : ] + binEdges1[ : -1]) / 2, binCounts1, width = binEdges1[1] - binEdges1[0], color = cm.cividis(0), label = "mean: " + str(numpy.round(numpy.mean(inSameJackknifeClusterFractionClusterUnique), 2)) + r" $\vert$ median: " + str(numpy.round(numpy.median(inSameJackknifeClusterFractionClusterUnique), 2)), capstyle = "round")
        axesGrid[indexCluster, 1].imshow(inSameJackknifeClusterFractionCluster, cmap = cm.cividis, vmin = 0, vmax = 1)

        correctRelationFractionCluster              = correctRelationFraction[indicesObservations]
        binCounts2, binEdges2                       = numpy.histogram(correctRelationFractionCluster, bins = binEdges2)
        axesGrid[indexCluster, 2].bar((binEdges2[1 : ] + binEdges2[ : -1]) / 2, binCounts2, width = binEdges2[1] - binEdges2[0], color = "gray", label = "mean: " + str(numpy.round(numpy.mean(correctRelationFractionCluster), 2)) + r" $\vert$ median: " + str(numpy.round(numpy.median(correctRelationFractionCluster), 2)))

        axesGrid[indexCluster, 0].grid(ls = "--", axis = "y", alpha = .2)
        axesGrid[indexCluster, 2].grid(ls = "--", axis = "y", alpha = .2)
        axesGrid[indexCluster, 0].legend(loc = "upper left")
        axesGrid[indexCluster, 2].legend(loc = "upper left")
        axesGrid[indexCluster, 0].set_xlim(binEdges1[0], binEdges1[-1])
        axesGrid[indexCluster, 2].set_xlim(binEdges2[0], binEdges2[-1])
        axesGrid[indexCluster, 1].set_xticks([])
        axesGrid[indexCluster, 1].set_yticks([])
        axesGrid[indexCluster, 0].set_xlabel("probability of staying together in jackknife clusterings (1)")
        axesGrid[indexCluster, 2].set_xlabel("correct clustering relation fraction (1)")
        axesGrid[indexCluster, 0].set_ylabel(r"\textnumero\ pairs (1)")
        axesGrid[indexCluster, 2].set_ylabel(r"\textnumero\ observations (1)")
        axesGrid[indexCluster, 0].set_title("cluster " + str(indexCluster + 1) + " ($N$ = " + str(numberOfObservationsCluster) + ")")
        axesGrid[indexCluster, 1].set_title("cluster " + str(indexCluster + 1) + " ($N$ = " + str(numberOfObservationsCluster) + ")")
        axesGrid[indexCluster, 2].set_title("cluster " + str(indexCluster + 1) + " ($N$ = " + str(numberOfObservationsCluster) + ")")

    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "Jackknife" + str(numberOfObservationsSubset) + figureExtension)
    pyplot.close()

'''
wrongRelationFraction          = numpy.sum(inSameCluster * (inSameJackknifeSubsetCount - inSameJackknifeClusterCount) + numpy.logical_not(inSameCluster) * inSameJackknifeClusterCount, axis = 0) / (numpy.sum(inSameJackknifeSubsetCount, axis = 0) - numpy.diag(inSameJackknifeSubsetCount))

for i in range(1078):
    print(correctRelationFraction[i] + wrongRelationFraction[i])
'''




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.

Reorders (or relabels) the clusters found in a clustering run to facilitate a particular interpretation.
Typically, this is run after a few plots have been made using 'AHCResampling', 'AHCResultsVisualisation', 'AHCResultsJackknife' and 'AHCResultsRawData'.
Be sure to rerun those scripts after reordering to generate up-to-date plots.

Note that this script does not automatically change the ordering of related clustering run outcomes (e.g. for the same linkage type, but for a slightly higher or lower cluster number).
'''
import numpy

def AHCReorder(directoryData,
               linkageType,
               indicesNew,
               dataSetName = "full"
               ):
    '''
    '''
    linkageTypeCapitalised            = linkageType.capitalize()
    numberOfClusters                  = len(indicesNew)
    nameFileClusterList               = "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList"               + ".npy"
    nameFileDistanceIntraClustersList = "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "DistanceIntraClustersList" + ".npy"

    # Load data.
    # Load the cluster list.
    clusterList                       = numpy.load(directoryData + dataSetName + "/" + nameFileClusterList, allow_pickle = True)
    # Load the between-clusters distance matrices.
    '''
    # Not currently implemented.
    '''
    # Load the within-cluster distances.
    distanceIntraClustersList         = numpy.load(directoryData + dataSetName + "/" + nameFileDistanceIntraClustersList, allow_pickle = True)

    # Create new lists.
    clusterListNew                    = []
    distanceIntraClustersListNew      = []
    for indexNew in indicesNew:
        clusterListNew.append(clusterList[indexNew])
        distanceIntraClustersListNew.append(distanceIntraClustersList[indexNew])

    # Save the results by overwriting previous files. By applying the appropriate permutation, this can be undone.
    numpy.save(directoryData + dataSetName + "/" + nameFileClusterList,               clusterListNew)
    numpy.save(directoryData + dataSetName + "/" + nameFileDistanceIntraClustersList, distanceIntraClustersListNew)




'''
Martijn Simon Soen Liong Oei, April 12021 H.E.
'''

import numpy, pandas

def AHCResultsAmendCSV(directoryData,
                       fileNameInput,
                       fileNameOutput,
                       linkageType,
                       numberOfClusters,
                       dataSetName = "full"
                       ):
    '''
    Add the cluster index for each observation to a CSV file, as a column.
    The CSV file to amend is 'fileNameInput', whilst the amended CSV file will be stored under 'fileNameOutput'.
    '''

    linkageTypeCapitalised = linkageType.capitalize()

    # Load data.
    clusterList            = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList" + ".npy", allow_pickle = True)
    numberOfObservations   = 0
    for cluster in clusterList:
        numberOfObservations += len(cluster)

    # Generate column.
    indicesCluster         = numpy.empty(numberOfObservations, dtype = int)
    for indexCluster in range(numberOfClusters):
        indicesCluster[clusterList[indexCluster]] = indexCluster + 1

    # Store column.
    dataFrame              = pandas.read_csv(directoryData + fileNameInput)
    dataFrame["cluster index (" + dataSetName + ", " + linkageType + ", N = " + str(numberOfClusters) + ")"] = indicesCluster
    dataFrame.to_csv(path_or_buf = directoryData + fileNameOutput, index = False)
