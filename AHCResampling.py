'''
Martijn Simon Soen Liong Oei, April 12020 H.E.

Calculates and visualises the non-randomness of the agglomerative hierarchical clustering by means of a resampling simulation.

clusterList:  numpy.ndarray of lists, contains cluster observation indices
clusterSizes: numpy.ndarray of ints,  contains cluster sizes
'''

from matplotlib import pyplot
import matplotlib, numpy, os
matplotlib.rcParams["text.usetex"] = True

def AHCResampling(directoryData,
                  directoryFigures,
                  dataSetName,
                  linkageType,
                  numberOfClusters,
                  dimensionsUsed,
                  listGroupNames,                   # names   of the dimension groups
                  listGroupIndices,                 # indices of the dimension groups
                  numberOfResamples       = 1000,   # number of random clusterings to simulate
                  numberOfObservationsMin = 15,     # number of observations required for a cluster on a dimension to show the significance
                  numberOfSigmata         = 6,      # number of standard deviations that should correspond with the extremes of the colour map
                  rowSpanGroups           = 2,      # number of rows occupied by the names of grouped dimensions
                  rowSpanDimensions       = 10,     # number of rows occupied by the names of dimensions
                  rowSpanCluster          = 5,      # number of rows occupied by a cluster
                  rowSpanWhiteSpace       = 1,      # number of rows occupied by the whitespace between the clusters and the colour bar
                  rowSpanColourBar        = 1,      # number of rows occupied by the colour bar
                  numberOfColumns         = 3,      # number of columns for the whole plot; used to control the colour bar length
                  plotScattersExtension   = ".pdf", # file format of scatter plots (".pdf" or ".png")
                  plotBarcodesExtension   = ".pdf", # file format of barcode plots (".pdf" or ".png")
                  fontSizeGroups          = 12,
                  fontSizeDimensions      = 12,
                  fontSizeClusters        = 12,
                  plotScatters            = True,
                  plotBarcodes            = True):
    '''
    '''

    linkageTypeCapitalised     = linkageType.capitalize()

    # Load data.
    # Load 'observationMatrix'.
    observationMatrix          = numpy.load(directoryData + dataSetName + "/" + "observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape

    nameFileOutput             = "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList" + ".npy"
    clusterList                = numpy.load(directoryData + dataSetName + "/" + nameFileOutput, allow_pickle = True)
    clusterSizes               = numpy.empty(numberOfClusters, dtype = int)
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
            pyplot.figure(figsize = (10, 8))

            pyplot.scatter(range(numberOfDimensions), numpy.nanmean(observationMatrix[clusterList[i]], axis = 0), c = "mediumseagreen", zorder = 2)
            for j in range(numberOfResamples):
                pyplot.scatter(range(numberOfDimensions), clusterResampledMeans[i][j], c = "gray", alpha = .05, zorder = 1)

            pyplot.scatter(range(numberOfDimensions), clusterResampledMeansMeans[i], facecolors = "none", edgecolors = "black")
            pyplot.xlim(-1, numberOfDimensions)
            pyplot.ylim(-0.05, 1.05)

            pyplot.gca().set_xticks(range(numberOfDimensions))
            pyplot.gca().set_xticklabels(dimensionsUsed, rotation = 90, fontsize = 6)

            pyplot.title("cluster " + str(i + 1))
            pyplot.grid(ls = "--", alpha = .2)
            pyplot.tight_layout()
            pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterSignificance" + str(i + 1) + plotScattersExtension)
            pyplot.close()


    if (plotBarcodes):
        figure          = pyplot.figure(figsize = (10, 8))

        numberOfRows    = rowSpanGroups + rowSpanDimensions + numberOfClusters * rowSpanCluster + rowSpanWhiteSpace + rowSpanColourBar

        axesGroups      = pyplot.subplot2grid((numberOfRows, numberOfColumns), (0, 0), rowspan = rowSpanGroups, colspan = 3)
        groups          = numpy.zeros(numberOfDimensions)
        for i in range(len(listGroupIndices) - 1):
            groups[listGroupIndices[i] : ] += 1
            axesGroups.text((listGroupIndices[i] + listGroupIndices[i + 1] - 1) / 2, 0, listGroupNames[i], horizontalalignment = "center", verticalalignment = "center", fontsize = fontSizeGroups)
        axesGroups.imshow(numpy.reshape(groups, (1, numberOfDimensions)), cmap = "Spectral", aspect = "auto", alpha = .75)
        axesGroups.set_xticks([])
        axesGroups.set_yticks([])

        axesDimensions  = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups, 0), rowspan = rowSpanDimensions, colspan = 3)
        axesDimensions.imshow(numpy.reshape(groups, (1, numberOfDimensions)), cmap = "Spectral", aspect = "auto", alpha = .5)
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
            axesCluster.imshow(significances, cmap = "coolwarm", aspect = "auto", vmin = -1 * numberOfSigmata, vmax = numberOfSigmata, alpha = 1)

            axesCluster.set_xticks([])
            axesCluster.set_yticks([0])
            axesCluster.set_yticklabels([r"\textbf{Cluster\ " + str(clusterIndex + 1) + r"}" + "\n" + r"$N = " + str(clusterSizes[clusterIndex]) + r"$"], fontsize = fontSizeClusters)
            axesCluster.tick_params(axis = "y", length = 0)

        axesWhiteSpace  = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + numberOfClusters * rowSpanCluster, 0), rowspan = rowSpanWhiteSpace, colspan = 3)
        axesWhiteSpace.set_visible(False)

        axesColourBar   = pyplot.subplot2grid((numberOfRows, numberOfColumns), (rowSpanGroups + rowSpanDimensions + numberOfClusters * rowSpanCluster + rowSpanWhiteSpace, 2), rowspan = rowSpanColourBar, colspan = 1)
        numberOfColours = 1000 + 1
        axesColourBar.imshow(numpy.reshape(numpy.linspace(-1 * numberOfSigmata, numberOfSigmata, num = numberOfColours, endpoint = True), (1, numberOfColours)), cmap = "coolwarm", aspect = "auto", vmin = -1 * numberOfSigmata, vmax = numberOfSigmata, alpha = 1)
        axesColourBar.set_xticks([-0.5, (numberOfColours - 1) / 2 - numberOfColours / (2 * numberOfSigmata) * 2, (numberOfColours - 1) / 2, (numberOfColours - 1) / 2 + numberOfColours / (2 * numberOfSigmata) * 2, numberOfColours - 0.5])
        axesColourBar.set_xticklabels([r"$-" + str(numberOfSigmata) + r"\ \sigma$", r"$-2\ \sigma$", r"$0$", r"$2\ \sigma$", r"$" + str(numberOfSigmata) + r"\ \sigma$"])
        axesColourBar.set_yticks([])

        figure.set_size_inches(w = 11.7 * 2, h = 8.3 * 2 * .95)
        pyplot.subplots_adjust(left = .04, right = .99, bottom = .02, top = .99)
        figure.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterSignificances" + plotBarcodesExtension)

        pyplot.close()
