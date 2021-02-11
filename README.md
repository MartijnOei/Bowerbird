# Bowerbird
Bowerbird is a Python package that performs agglomerative hierarchical clustering (AHC) with smart statistical follow-up analyses. It also generates publication-ready visuals.

![plot](https://github.com/MartijnOei/Bowerbird/blob/main/logoBowerbirdLarge.png)

# Install

# Quick start
A code snippet that performs AHC is:
```python
import AHCCompute
directoryData               = "./data/"
dataSetName                 = "full"
linkageType                 = "complete"
numberOfDimensions          = 10
dimensionalWeights          = [1] * numberOfDimensions
numberOfClustersStartSaving = 10

AHCCompute.AHCCompute(directoryData, dataSetName, linkageType, dimensionalWeights, numberOfClustersStartSaving)
```
