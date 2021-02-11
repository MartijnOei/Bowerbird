![plot](https://github.com/MartijnOei/Bowerbird/blob/main/logoBowerbirdLarge.png)

# Bowerbird
Bowerbird is a Python package that performs agglomerative hierarchical clustering (AHC) with smart statistical follow-up analyses. It also generates publication-ready visuals.

Bowerbirds, or _Ptilonorhynchidae,_ display complex courtship behaviour. In order to attract a mate, they form clusters of objects with similar properties (colours, mostly).

# Install

# Quick start
Bowerbird's power is unleashed with a few successive function calls.
To prepare your full data set, e.g. use:
```python
import Bowerbird
directoryData               = "./data/"
fileName                    = "dataSetExample.csv"
indexColumnStart            = 0
indexColumnEnd              = 5
dataSetName                 = "full"
Bowerbird.AHCPrepareDataSetFull(directoryData, fileName, indexColumnStart, indexColumnEnd, dataSetName = dataSetName)
```
To perform AHC, e.g. add:
```python
linkageType                 = "complete"
numberOfDimensions          = 5
dimensionalWeights          = [1] * numberOfDimensions
numberOfClustersStartSaving = 10

AHCCompute.AHCCompute(directoryData, dataSetName, linkageType, dimensionalWeights, numberOfClustersStartSaving)
```
