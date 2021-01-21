'''
Martijn Simon Soen Liong Oei, April 12020 H.E.

This program performs agglomerative hierarchical clustering using Gower's distance.
The data consists of 'numberOfObservations' vectors of length 'numberOfDimensions', and is allowed to have missing entries.
'''

import AHCFunctions, numpy, time

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
               dataSetName,
               linkageType,                # all options: "complete", "average", "single"
               dimensionalWeights,
               numberOfClustersStartSaving # the highest number of clusters for which data is saved (... and the lowest being fixed to 2)
               ):
    '''
    '''

    # Load data.
    observationMatrix                   = numpy.load(directoryData + dataSetName + "/observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape
    hasObservation                      = AHCFunctions.loadAvailabilityMatrix(observationMatrix)
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

    distanceInterClustersMatrix         = AHCFunctions.distancesGower(observationMatrix, hasObservation, dimensionalWeights, range(numberOfObservations), range(numberOfObservations)).reshape((numberOfObservations, numberOfObservations))
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
            distances                     = AHCFunctions.distancesGower(observationMatrix, hasObservation, dimensionalWeights, clusterOld, clusterNew)

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
        distanceIntraClusterNew             = numpy.concatenate((distanceIntraClustersList[indexClusterA], distanceIntraClustersList[indexClusterB], AHCFunctions.distancesGower(observationMatrix, hasObservation, dimensionalWeights, clusterA, clusterB)))
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
            silhouettesAll                    = AHCFunctions.silhouettes(observationMatrix, hasObservation, dimensionalWeights, clusterList)
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
                distances             = AHCFunctions.distancesToClusterMean(observationMatrix[clusterList[i]], hasObservation[clusterList[i]], dimensionalWeights)
                varianceModelCurrent += numpy.sum(numpy.square(distances))

            distances                         = AHCFunctions.distancesToClusterMean(observationMatrix, hasObservation, dimensionalWeights)
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
        print (clusterIndicesList)

    dendrogramMatrix[numberOfObservations - 2, 0] = clusterIndicesList[0]
    dendrogramMatrix[numberOfObservations - 2, 1] = clusterIndicesList[1]
    dendrogramMatrix[numberOfObservations - 2, 2] = 1
    dendrogramMatrix[numberOfObservations - 2, 3] = numberOfObservations
    numpy.save(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + "DendrogramMatrix.npy", dendrogramMatrix)
