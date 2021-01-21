'''
Martijn Simon Soen Liong Oei, April 12020 H.E.

This program compares the coefficients of determination obtained with clustering of uniform random data to clustering of actual data.
'''

from matplotlib import pyplot
import matplotlib, numpy
matplotlib.rcParams["text.usetex"] = True

def AHCResultsTestUniformity(directoryData,
                             directoryFigures,
                             linkageType,
                             numberOfDataSets,
                             numberOfNumerals,
                             numberOfClustersHighest,
                             numberOfClustersLowest,
                             colourActual  = "cornflowerblue",
                             colourUniform = "crimson",
                             alphaActual   = .3,
                             alphaUniform  = .3):
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
    pyplot.figure(figsize = (6, 4))

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
    pyplot.xticks(rangeClusterNumberMetrics, fontsize = "x-large")
    pyplot.yticks(fontsize = "x-large")
    pyplot.xlim(rangeClusterNumberMetrics[0] + .1, rangeClusterNumberMetrics[-1] - .1)
    pyplot.ylim(0, 50)
    pyplot.xlabel("number of clusters (1)", fontsize = "x-large")
    pyplot.ylabel(r"explained variance (\%)", fontsize = "x-large")
    pyplot.legend(loc = "upper right", fontsize = "x-large", borderpad = .15)
    #pyplot.title(r"goodness of fit $\vert$ coefficient of determination $R^2$")
    pyplot.tight_layout()
    pyplot.savefig(directoryFigures + "AHC" + linkageTypeCapitalised + "CoefficientsOfDeterminationSignificance" + ".pdf")
    pyplot.close()
