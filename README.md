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

# Follow-up analysis
Bowerbird is especially strong at performing statistical stress tests of the clustering result.

Bowerbird uses a resampling method to explore, for each cluster, on which dimensions significant deviations occur from the total population.
To generate the resampling output for a 4-cluster scenario:
```python
numberOfClusters            = 4
Bowerbird.AHCResampling(directoryData, directoryFigures, dataSetName, linkageType, numberOfClusters, dimensionsUsed, listGroupNames, listGroupIndices)
```
