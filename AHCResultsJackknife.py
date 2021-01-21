'''
Martijn Simon Soen Liong Oei, April 12020 H.E.
'''

import numpy
from scipy import stats
from matplotlib import pyplot
import matplotlib
matplotlib.rcParams["text.usetex"] = True


def AHCResultsJackknife(directoryData, directoryFigures, linkageType, numberOfClusters, numberOfDataSets, numberOfObservationsSubset, numberOfNumerals, probabilityMixingMin = 0, probabilityMixingMax = .5, colourMapMixing = "magma", colourTextMixing = "white", showRanks = True):
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
    pyplot.figure(figsize = (8, 6)) #(12, 9)
    pyplot.imshow(mixingMatrix, cmap = colourMapMixing, vmin = probabilityMixingMin, vmax = probabilityMixingMax)
    objectColourBar = pyplot.colorbar()#, size = "x-large")#, labelsize = "x-large")
    objectColourBar.ax.tick_params(labelsize = 22)#"xx-large")
    objectColourBar.set_label(label = "mixing probability (1)", size = 22)#"xx-large")

    for i in range(numberOfClusters):
        for j in range(i + 1, numberOfClusters):
            if (showRanks):
                pyplot.text(j, i, r"\textbf{rank " + str(mixingMatrixRank[i, j]) + ":}\n" + str(int(numpy.round(mixingMatrix[i, j] * 100))) + r"\%", c = colourTextMixing, horizontalalignment = "center", verticalalignment = "center", fontsize = "x-large")
            else:
                pyplot.text(j, i, str(int(numpy.round(mixingMatrix[i, j] * 100))) + r"\%", c = colourTextMixing, horizontalalignment = "center", verticalalignment = "center", fontsize = "x-large")
    #pyplot.xlabel("full data set cluster ID")
    #pyplot.ylabel("full data set cluster ID")
    pyplot.xlabel("cluster", fontsize = 22)#"xx-large")
    pyplot.ylabel("cluster", fontsize = 22)#"xx-large")
    pyplot.gca().set_xticks(numpy.arange(numberOfClusters))
    pyplot.gca().set_xticklabels(numpy.arange(numberOfClusters) + 1)
    pyplot.gca().set_yticks(numpy.arange(numberOfClusters))
    pyplot.gca().set_yticklabels(numpy.arange(numberOfClusters) + 1)
    pyplot.xticks(fontsize = 22)#"xx-large")
    pyplot.yticks(fontsize = 22)#"xx-large")
    #pyplot.title(r"\textbf{agglomerative hierarchical clustering} jackknife analysis" + "\n" + linkageType + r" linkage $\vert$ \textnumero\ clusters: " + str(numberOfClusters) + r" $\vert$ \textnumero\ observations in subset: " + str(numberOfObservationsSubset))
    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "Jackknife" + str(numberOfObservationsSubset) + "Mixing.pdf")
    pyplot.close()


    pyplot.figure(figsize = (12, 9))
    pyplot.imshow(inSameJackknifeClusterFraction, cmap = "cividis", vmin = 0, vmax = 1)
    pyplot.colorbar(label = "probability of staying together in jackknife clusterings (1)")
    pyplot.xlabel("full data set observation index (1)")
    pyplot.ylabel("full data set observation index (1)")
    pyplot.title(r"\textbf{agglomerative hierarchical clustering} jackknife analysis" + "\n" + linkageType + r" linkage $\vert$ \textnumero\ clusters: " + str(numberOfClusters) + r" $\vert$ \textnumero\ observations in subset: " + str(numberOfObservationsSubset))
    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "Jackknife" + str(numberOfObservationsSubset) + "All.pdf")
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

        axesGrid[indexCluster, 0].bar((binEdges1[1 : ] + binEdges1[ : -1]) / 2, binCounts1, width = binEdges1[1] - binEdges1[0], color = "navy", label = "mean: " + str(numpy.round(numpy.mean(inSameJackknifeClusterFractionClusterUnique), 2)) + r" $\vert$ median: " + str(numpy.round(numpy.median(inSameJackknifeClusterFractionClusterUnique), 2)))
        axesGrid[indexCluster, 1].imshow(inSameJackknifeClusterFractionCluster, cmap = "cividis", vmin = 0, vmax = 1)

        correctRelationFractionCluster              = correctRelationFraction[indicesObservations]
        binCounts2, binEdges2                       = numpy.histogram(correctRelationFractionCluster, bins = binEdges2)
        axesGrid[indexCluster, 2].bar((binEdges2[1 : ] + binEdges2[ : -1]) / 2, binCounts2, width = binEdges2[1] - binEdges2[0], color = "gray", label = "mean: " + str(numpy.round(numpy.mean(correctRelationFractionCluster), 2)) + r" $\vert$ median: " + str(numpy.round(numpy.median(correctRelationFractionCluster), 2)))

        axesGrid[indexCluster, 0].grid(ls = "--", axis = "y", alpha = .2)
        axesGrid[indexCluster, 2].grid(ls = "--", axis = "y", alpha = .2)
        axesGrid[indexCluster, 0].legend(loc = "upper right")
        axesGrid[indexCluster, 2].legend(loc = "upper right")
        axesGrid[indexCluster, 0].set_xlim(binEdges1[0], binEdges1[-1])
        axesGrid[indexCluster, 2].set_xlim(binEdges2[0], binEdges2[-1])
        axesGrid[indexCluster, 1].set_xticks([])
        axesGrid[indexCluster, 1].set_yticks([])
        axesGrid[indexCluster, 0].set_xlabel("probability of staying together in jackknife clusterings (1)")
        axesGrid[indexCluster, 2].set_xlabel("correct clustering relation fraction (1)")
        axesGrid[indexCluster, 0].set_ylabel("number of pairs (1)")
        axesGrid[indexCluster, 2].set_ylabel("number of observations (1)")
        axesGrid[indexCluster, 0].set_title("cluster " + str(indexCluster + 1) + " ($N$ = " + str(numberOfObservationsCluster) + ")")
        axesGrid[indexCluster, 1].set_title("cluster " + str(indexCluster + 1) + " ($N$ = " + str(numberOfObservationsCluster) + ")")
        axesGrid[indexCluster, 2].set_title("cluster " + str(indexCluster + 1) + " ($N$ = " + str(numberOfObservationsCluster) + ")")

    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageType.capitalize() + str(numberOfClusters) + "Jackknife" + str(numberOfObservationsSubset) + ".pdf")
    pyplot.close()

'''
wrongRelationFraction          = numpy.sum(inSameCluster * (inSameJackknifeSubsetCount - inSameJackknifeClusterCount) + numpy.logical_not(inSameCluster) * inSameJackknifeClusterCount, axis = 0) / (numpy.sum(inSameJackknifeSubsetCount, axis = 0) - numpy.diag(inSameJackknifeSubsetCount))

for i in range(1078):
    print(correctRelationFraction[i] + wrongRelationFraction[i])
'''
