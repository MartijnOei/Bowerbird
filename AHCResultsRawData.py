'''
Martijn Simon Soen Liong Oei, April 12020 H.E.
'''

from matplotlib import cm, pyplot
import AHCFunctions, matplotlib, numpy, os, pandas
matplotlib.rcParams["text.usetex"] = True

def AHCResultsRawData(directoryData,
                      directoryFigures,
                      fileName,
                      indexColumnStart,
                      indexColumnEnd,
                      dataSetName,
                      linkageType,
                      numberOfClusters,
                      numberOfDimensionsPerFigure = 20,
                      colourMap                   = cm.Spectral,
                      faceColour                  = "0.8",
                      numberOfBins                = 10,
                      colourBinText               = "white"):
    '''
    '''

    linkageTypeCapitalised      = linkageType.capitalize()

    # Load data.
    clusterList                 = numpy.load(directoryData + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClusterList" + ".npy", allow_pickle = True)
    observationMatrix, dimensionsUsed, numberOfObservations, numberOfDimensions = AHCFunctions.loadObservationMatrix(directoryData + fileName, indexColumnStart, indexColumnEnd)


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
                axes = pyplot.subplot2grid((numberOfClusters, numberOfDimensionsPerFigure), (indexCluster, indexDimension - indexFigure * numberOfDimensionsPerFigure), rowspan = 1, colspan = 1)
                observationsPlot = observationMatrix[clusterList[indexCluster], indexDimension]
                numberOfObservationsPlot = len(observationsPlot)
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
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + linkageTypeCapitalised + str(numberOfClusters) + "ClustersRawData" + str(indexFigure) + ".pdf")
        pyplot.close()
