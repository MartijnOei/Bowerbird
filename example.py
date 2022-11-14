import Bowerbird

if __name__ == "__main__":
    directoryData               = "./"                # path that contains the data set
    fileName                    = "BowerbirdMockData.csv"  # name of the CSV file
    indexColumnStart            = 1                        # index of the first column to be read out (0-based: so 1 refers to the second column)
    dataSetName                 = "full"                   # name given to the data set within Bowerbird; should be used in other function calls
    normalise                   = [True] * 6 + [False] * 3 # normalise the first 6 columns, but not the last 3
    Bowerbird.AHCPrepareDataSetFull(directoryData, fileName, indexColumnStart = indexColumnStart, dataSetName = dataSetName, normalise = normalise)

    linkageType                 = "complete"                     # all options: "complete", "average", "single"
    dimensionalWeights          = [.5] * 2 + [.5] * 2 + [.2] * 5 # use 3 groups of dimensions (body, courtship and vocals), and give each group equal importance by dividing weight 1 over the group's dimensions
    numberOfClustersStartSaving = 15                             # save clustering output from this cluster number up to and including 2 (the 1-cluster result is trivial)
    Bowerbird.AHCCompute(directoryData, linkageType, dimensionalWeights, numberOfClustersStartSaving, dataSetName = dataSetName)

    # Fun fact: some bowerbirds are masters of mimicry, emulating the sounds of pigs, humans and... waterfalls.
    directoryFigures            = "./figures/" # path used to store figures (doesn't need to exist yet)
    numberOfClustersHighest     = 15           # highest cluster number to show in progression figures
    numberOfClustersLowest      = 2            # lowest  cluster number to show in progression figures
    dimensionsUsed              = ["body length (cm)", "body mass (g)", "court area (sq. m)", "number of partners\nthis year (1)", "mean\nsong duration (s)", "mean\nsong loudness (dB)", "mimics pigs", "mimics humans", "mimics waterfalls"] # names of dimensions
    listGroupNames              = [r"\textbf{body}", r"\textbf{courtship}", r"\textbf{vocals}"] # names of groups
    listGroupIndices            = [0, 2, 4]    # indices of each group's first dimension
    Bowerbird.AHCResultsVisualisation(directoryData, directoryFigures, linkageType, numberOfClustersHighest, numberOfClustersLowest, dimensionsUsed, listGroupNames, listGroupIndices, dataSetName = dataSetName)