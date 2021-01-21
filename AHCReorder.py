'''
Martijn Simon Soen Liong Oei, June 12020 H.E.

Reorders (or relabels) the clusters found in a clustering run to facilitate a particular interpretation.
Typically, this is run after a few plots have been made using 'AHCResampling', 'AHCResultsVisualisation', 'AHCResultsJackknife' and 'AHCResultsRawData'.
Be sure to rerun those scripts after reordering to generate up-to-date plots.

Note that this script does not automatically change the ordering of related clustering run outcomes (e.g. for the same linkage type, but for a slightly higher or lower cluster number).
'''
import numpy

def AHCReorder(directoryData, dataSetName, linkageType, indicesNew):
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
