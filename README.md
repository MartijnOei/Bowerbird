![plot](https://github.com/MartijnOei/Bowerbird/blob/main/logoBowerbirdLarge.png)

# Bowerbird
Bowerbird is a Python package that performs agglomerative hierarchical clustering (AHC) with smart statistical follow-up analyses. It also generates publication-ready visuals. A scientific publication in which Bowerbird is used, is (Gool _et al.,_ 12021 H.E.).

Bowerbirds, or _Ptilonorhynchidae,_ are known for complex courtship behaviour. :heart: In order to attract a mate, they form clusters of objects with similar properties (colours, mostly), which are put on display in their well-kept jungle courts. :palm_tree: :seedling:

If you are interested in using Bowerbird for a scientific publication, or if you need help using it, please contact me! Feature suggestions are also welcome. :email:

# Install
Save ```Bowerbird.py``` to your project's code folder.

# Quick start
Bowerbird's power is unleashed with a few successive function calls. Here's an example with mock data set `dataSetPopulations.csv` (see repo files). Do these observations of some bowerbird species contain evidence for distinct subpopulations (clusters)?

To prepare the full data set:
```python
import Bowerbird
directoryData               = "./data/"
fileName                    = "dataSetPopulations.csv"
indexColumnStart            = 0
indexColumnEnd              = 5
dataSetName                 = "full"
Bowerbird.AHCPrepareDataSetFull(directoryData, fileName, indexColumnStart, indexColumnEnd, dataSetName = dataSetName)
```
To perform AHC:
```python
linkageType                 = "complete"
numberOfDimensions          = 5
dimensionalWeights          = [1] * numberOfDimensions
numberOfClustersStartSaving = 10

Bowerbird.AHCCompute(directoryData, dataSetName, linkageType, dimensionalWeights, numberOfClustersStartSaving)
```
To visualise results:
```python
# Fun fact: some bowerbirds are masters of mimicry, emulating the sounds of pigs, humans and... waterfalls.
directoryFigures            = "./figures/"
numberOfClustersHighest     = 10
numberOfClustersLowest      = 3
dimensionsUsed              = ["body length (cm)", "body weight (g)", "court area (sq. m)", "mean song duration (s)", "waterfall mimicry"]
listGroupNames              = [r"\textbf{body}", r"\textbf{bower}", r"\textbf{vocals}"]
listGroupIndices            = [0, 2, 3]
Bowerbird.AHCResultsVisualisation(directoryData, directoryFigures, dataSetName, linkageType, numberOfClustersHighest, numberOfClustersLowest, dimensionsUsed, listGroupNames, listGroupIndices)
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
However, there exist different notions of inter-cluster distance: there is _complete_, _average_ and _single_ linkage. To see how the distances between the clusters differ under other notions of inter-cluster distance, Bowerbird also provides inter-cluster distance overviews for the other linkage types than the one used for arriving at the clustering result. For example, the _average_ linkage inter-cluster distances for the same clusters, are:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5DistanceInterClustersMatrixAverage.png)
Interestingly, under this definition, cluster 1 and 5 are most nearby. The choice of linkage type really matters! However, note that cluster 2 and 4 remain the farthest apart.

# Follow-up analysis
Bowerbird is especially strong at performing statistical tests of the clustering result.

For example, along which dimensions do the clusters deviate significantly from the population as a whole? To answer this question, Bowerbird uses a resampling technique that generates random clusters (of exactly the same sizes as the original clusters, however), and compares the original cluster centres to the random cluster centres. To resample:
```python
numberOfClusters            = 5
Bowerbird.AHCResampling(directoryData, directoryFigures, linkageType, numberOfClusters, dimensionsUsed, listGroupNames, listGroupIndices, dataSetName = dataSetName)
```
This yields the following result for cluster 2, where the green dots represent the original cluster, and the grey dots the random ones:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5ClusterSignificance2.png)
The function above also outputs the resampling results for _all_ clusters in a single figure, where deviations between 'green' and 'grey' (in multiples of the standard deviation of the grey points) are shown on a colour scale:
![plot](https://github.com/MartijnOei/Bowerbird/blob/main/AHCComplete5ClusterSignificances.png)

<!---
%Bowerbird uses a resampling method to explore, for each cluster, on which dimensions significant deviations occur from the total population.
%To generate the resampling output for a 4-cluster scenario:
```python
numberOfClusters            = 4
Bowerbird.AHCResampling(directoryData, directoryFigures, dataSetName, linkageType, numberOfClusters, dimensionsUsed, listGroupNames, listGroupIndices)
```
--->
