![plot](https://github.com/MartijnOei/Bowerbird/blob/main/logoBowerbirdLarge.png)

# Bowerbird
Bowerbird is a Python package that performs agglomerative hierarchical clustering (AHC) with smart statistical follow-up analyses. It also generates publication-ready visuals. A scientific publication in which Bowerbird is used, is (Gool _et al.,_ 12021 H.E.).

Bowerbirds, or _Ptilonorhynchidae,_ are known for complex courtship behaviour. :heart: In order to attract a mate, they form clusters of objects with similar properties (colours, mostly), which are put on display in their well-kept jungle courts. :palm_tree: :seedling:

If you are interested in using Bowerbird for a scientific publication, or if you need help using it, please contact me! Feature suggestions are also welcome. :email:

# Install
Save ```Bowerbird.py``` to your project's code folder.

Bowerbird's dependencies (apart from the [Python Standard Library](https://docs.python.org/3/library/ "https://docs.python.org/3/library/")) are ```matplotlib```, ```numpy```, ```pandas``` and ```scipy```.

# Quick start
Bowerbird's power is unleashed with a few successive function calls. Here's an example with mock data set `dataSetPopulations.csv` (see repo files). Do these observations of some bowerbird species contain evidence for distinct subpopulations (clusters)?

To prepare the full data set:
```python
import Bowerbird

directoryData               = "./data/"                # path that contains the data set
fileName                    = "BowerbirdMockData.csv"  # name of the CSV file
indexColumnStart            = 1                        # index of the first column to be read out (0-based: so 1 refers to the second column)
dataSetName                 = "full"                   # name given to the data set within Bowerbird; should be used in other function calls
normalise                   = [True] * 6 + [False] * 3 # normalise the first 6 columns, but not the last 3
Bowerbird.AHCPrepareDataSetFull(directoryData, fileName, indexColumnStart = indexColumnStart, dataSetName = dataSetName, normalise = normalise)
```
To perform AHC:
```python
linkageType                 = "complete"                     # all options: "complete", "average", "single"
dimensionalWeights          = [.5] * 2 + [.5] * 2 + [.2] * 5 # use 3 groups of dimensions (body, courtship and vocals), and give each group equal importance by dividing weight 1 over the group's dimensions
numberOfClustersStartSaving = 15                             # save clustering output from this cluster number up to and including 2 (the 1-cluster result is trivial)
Bowerbird.AHCCompute(directoryData, linkageType, dimensionalWeights, numberOfClustersStartSaving, dataSetName = dataSetName)
```
To visualise results:
```python
# Fun fact: some bowerbirds are masters of mimicry, emulating the sounds of pigs, humans and... waterfalls.
directoryFigures            = "./figures/" # path used to store figures (doesn't need to exist yet)
numberOfClustersHighest     = 15           # highest cluster number to show in progression figures
numberOfClustersLowest      = 2            # lowest  cluster number to show in progression figures
dimensionsUsed              = ["body length (cm)", "body mass (g)", "court area (sq. m)", "number of partners\nthis year (1)", "mean\nsong duration (s)", "mean\nsong loudness (dB)", "mimics pigs", "mimics humans", "mimics waterfalls"] # names of dimensions
listGroupNames              = [r"\textbf{body}", r"\textbf{courtship}", r"\textbf{vocals}"] # names of groups
listGroupIndices            = [0, 2, 4]    # indices of each group's first dimension
Bowerbird.AHCResultsVisualisation(directoryData, directoryFigures, linkageType, numberOfClustersHighest, numberOfClustersLowest, dimensionsUsed, listGroupNames, listGroupIndices, dataSetName = dataSetName)
```

One way to judge the quality of the result is to calculate the silhouette (higher is better) for each bird. This is how the mean and standard deviation of the silhouettes change as a function of the number of clusters:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete2ProgressionSilhouettes.png)
Each cluster has a centre, calculated simply by taking the arithmetic average of its members' parameter vectors. The cluster centre acts as an 'archetype' for that cluster, and one could argue that the goal of clustering is to find a small set of archetypes that describe the whole data set as accurate as possible. In this sense, clustering can be seen as a parameter fitting procedure. This is how the coefficient of determination changes as a function of the number of clusters:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete2CoefficientsOfDetermination.png)

The mean silhouette curve shows a suspicious bump at 5 clusters, and the coefficient of determination rapidly decays just after 5 clusters. This suggests that the data set might contain 5 clusters. Bowerbird allows us to inspect the 5 cluster result in great detail. For example, we can visualise individual clusters:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5Cluster3.png)
Or we can visualise the cluster centres:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5ClusterMeans.png)
Moreover, this function visualises the continuation of the clustering process until all birds are unified in a single cluster:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5Dendrogram.png)
One can see that clusters 1 and 3 will merge during the next step of the clustering algorithm. This is decided based on the fact that the inter-cluster distance between clusters 1 and 3 is smaller than that of any other cluster pair. Bowerbird provides insight into the inter-cluster distances at any desired step:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5DistanceInterClustersMatrixComplete.png)
Indeed, the distance between clusters 1 and 3 is the smallest distance of all.
However, there exist different notions of inter-cluster distance: there is _complete_, _average_ and _single_ linkage. To see how the distances between the clusters differ under other notions of inter-cluster distance, Bowerbird also provides inter-cluster distance overviews for linkage types other than the one used to arrive at the actual clustering result. For example, the average linkage inter-cluster distances for the same clusters, are:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5DistanceInterClustersMatrixAverage.png)
Interestingly, under _this_ notion, cluster 1 and 5 are the nearest neighbours. The choice of linkage type really matters! (However, note that cluster 2 and 4 remain the farthest apart.)

The [silhouette](https://en.wikipedia.org/wiki/Silhouette_(clustering) "https://en.wikipedia.org/wiki/Silhouette_(clustering)") is a standard clustering performance measure, and the distribution of silhouettes per cluster are part of Bowerbird's standard output:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5SilhouettesAll.png)

Often, one normalises the data of each dimension to a common scale (from 0 to 1) before clustering. How does the clustering result look like for the unnormalised data? To answer this question, use:
```python
numberOfClusters            = 5 # cluster number to visualise the results for
Bowerbird.AHCResultsRawData(directoryData, directoryFigures, fileName, linkageType, numberOfClusters, indexColumnStart = indexColumnStart, dataSetName = dataSetName)
```
For the mock data set, this yields:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5ClustersRawData0.png)


# Follow-up analysis
Bowerbird is especially strong at performing statistical tests of the clustering result.

## Significance
For example, along which dimensions do the clusters deviate significantly from the population as a whole? To answer this question, Bowerbird uses a resampling technique that generates random clusters (of exactly the same sizes as the original clusters, however), and compares the original cluster centres to the random cluster centres. To resample:
```python
Bowerbird.AHCResampling(directoryData, directoryFigures, linkageType, numberOfClusters, dimensionsUsed, listGroupNames, listGroupIndices, dataSetName = dataSetName)
```
This yields the following result for cluster 2, where the green dots represent the original cluster, and the grey dots the random ones:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5ClusterSignificance2.png)
The function above also outputs the resampling results for _all_ clusters in a single figure, where deviations between 'green' and 'grey' (in multiples of the standard deviation of the grey points) are shown on a colour scale:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5ClusterSignificances.png)

## Intrinsic clustering
The critical reader might note that even if a data set does not contain _any_ intrinsic clustering (but instead represents a continuum), clustering algorithms - by design - will still suggest 'clusters' in the data that deviate significantly from the general population (and from eachother) on at least some dimensions. Despite this, running a clustering algorithm would in such case not be particularly meaningful.
Bowerbird contains functionality to compare the actual coefficient of determination curve to curves expected if the data set did not contain any intrinsic clustering. It does this by clustering data sets drawn from a uniform distribution over the parameter space.

To prepare uniform data sets:
```python
numberOfDataSets            = 20 # number of data sets to prepare (of the uniform type)
numberOfNumerals            = 3  # number of numerals used in naming data sets (2 would be enough for 20 data sets, as it allows for names '00' up to '99')
Bowerbird.AHCPrepareDataSetUniform(directoryData, fileName, numberOfDataSets, indexColumnStart = indexColumnStart, numberOfNumerals = numberOfNumerals)
```

To perform clustering on them:
```python
dataSetNames                = []
for indexDataSet in range(numberOfDataSets):
    dataSetNames.append("uniform" + "_" + str(indexDataSet).zfill(numberOfNumerals))

for dataSetName in dataSetNames:
    Bowerbird.AHCCompute(directoryData, linkageType, dimensionalWeights, numberOfClustersStartSaving, dataSetName = dataSetName)
    print("Finished clustering on data set '" + dataSetName + "'!")
```

To visualise the coefficient of determination curve comparison:
```python
Bowerbird.AHCResultsTestUniformity(directoryData, directoryFigures, linkageType, numberOfDataSets, numberOfNumerals, numberOfClustersHighest, numberOfClustersLowest)
```

This yields:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCCompleteCoefficientsOfDeterminationSignificance.png)
Clearly, the parameter vectors of the birds in the mock data set are not drawn from a uniform distribution over parameter space. There is intrinsic clustering.

## Jackknife
To what extent is the clustering result driven by the data of a few (outlier) birds? And have we already entered the sample size regime in which the clustering result has 'converged', or can we expect significant changes in the algorithm's outcome if we were to collect more data?
To answer these questions, Bowerbird contains jackknife routines. To prepare jackknife data sets, use:
```python
numberOfDataSets            = 50   # number of data sets to prepare (of the jackknife type)
numberOfObservationsSubset  = 1000 # number of observations per data set
Bowerbird.AHCPrepareDataSetJackknife(directoryData, fileName, numberOfDataSets, numberOfObservationsSubset, indexColumnStart = indexColumnStart, numberOfNumerals = numberOfNumerals)
```
In this example, each jackknife subset contains 1000 birds. The total data set contains 1423 birds. Each jackknife subset thus is a random 70% subset of the total data set.
To perform clustering on them:
```python
dataSetNames                = []
for indexDataSet in range(numberOfDataSets):
    dataSetNames.append("jackknife" + str(numberOfObservationsSubset) + "_" + str(indexDataSet).zfill(numberOfNumerals))

numberOfClustersStartSaving = 10 # to save disk space, we reduce the highest cluster number for which results are stored (if we only want to generate the 5-cluster jackknife result, we could set this to 5)
for dataSetName in dataSetNames:
    Bowerbird.AHCCompute(directoryData, linkageType, dimensionalWeights, numberOfClustersStartSaving, dataSetName = dataSetName)
    print("Finished clustering on data set '" + dataSetName + "'!")
```
To generate jackknife results:
```python
Bowerbird.AHCResultsJackknife(directoryData, directoryFigures, linkageType, numberOfClusters, numberOfDataSets, numberOfObservationsSubset, numberOfNumerals = numberOfNumerals)
```
When clustering a jackknife data set, some birds will be put in the same cluster although they 'actually' belong to different ones, if the full data set clustering is to be believed. For each pair of different full data set clusters, we consider all bird pairs with one bird from one cluster, and the other bird from the other. These birds shouldn't be in the same cluster in a jackknife run. We call the probability that the birds of a random such pair are clustered together in a jackknife run, the _mixing probability._ This gives rise to the following matrix:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5Jackknife1000Mixing.png)

Now take a particular full data set cluster: cluster 3, say. For each pair of birds in this cluster, we can calculate the fraction of jackknife runs (that contain the pair!) in which this pair was indeed clustered together. This _probability of staying together in jackknife clusterings_ is shown in the central column of the figure below. In the left column, we show these data collapsed into a probability distribution. In the right column, we zoom out further. For each of the cluster's observations, we calculate the fraction of relations the observation had (with _all_ other observations in the full data set) that have remained the same in jackknife runs. This yields the _correct clustering relation fraction._
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5Jackknife1000.png)

# Convenience
Bowerbird contains some additional convenience routines.

After running agglomerative hierarchical clustering and settling on a particular number of clusters, one might want to reorder the clusters found (by simply changing 'labels') to facilitate a particular interpretation (brought forth in e.g. the accompanying research paper). To swap the order of the last two clusters in the 5-cluster result of the full data set, use:
```python
indicesNew                  = [0, 1, 2, 4, 3] # indices of the current clusters arranged in their future order
Bowerbird.AHCReorder(directoryData, linkageType, indicesNew, dataSetName = "full")
```
Note that this routine does not automatically change the ordering of related results (such as the 4- or 6-cluster result).
And make sure to rerun the visualisation routines after reordering to generate up-to-date figures.

It can be useful to add each bird's cluster ID to the raw data for other (non-Bowerbird) analyses. This is done with a single line: 
```python
fileNameInput               = fileName                        # name of the CSV file (input)
fileNameOutput              = fileName[ : -4] + "Amended.csv" # name of the CSV file (output)
Bowerbird.AHCResultsAmendCSV(directoryData, fileNameInput, fileNameOutput, linkageType, numberOfClusters, dataSetName = "full")
```
<!---
%Bowerbird uses a resampling method to explore, for each cluster, on which dimensions significant deviations occur from the total population.
%To generate the resampling output for a 4-cluster scenario:
```python
numberOfClusters            = 4
Bowerbird.AHCResampling(directoryData, directoryFigures, dataSetName, linkageType, numberOfClusters, dimensionsUsed, listGroupNames, listGroupIndices)
```
--->
