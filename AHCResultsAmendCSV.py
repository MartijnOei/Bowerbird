'''
Martijn Simon Soen Liong Oei, April 12020 H.E.
'''

import numpy, pandas

def AHCResultsAmendCSV(directoryData, fileNameInput, fileNameOutput, dataSetName, linkageType, numberOfClusters):
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
