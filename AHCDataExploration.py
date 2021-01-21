'''
Martijn Simon Soen Liong Oei, April 12020 H.E.
'''

from matplotlib import cm, colorbar, pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import AHCFunctions, numpy, os
import matplotlib
matplotlib.rcParams["text.usetex"] = True


def AHCDataExploration(directoryData,
                       directoryFigures,
                       dataSetName,
                       dimensionsUsed,
                       plotObservationMatrix  = True,
                       plotAvailabilityMatrix = True,
                       projectName            = "",
                       colourBarWidth         = "2.5%",
                       colourBarDistance      = .06,
                       colourMap              = cm.coolwarm,
                       colourMapBinary        = cm.RdYlGn):
    '''
    '''

    colourMap.set_bad(color = "white")

    # Load 'observationMatrix' and 'hasObservation'.
    observationMatrix = numpy.load(directoryData + dataSetName + "/" + "observationMatrix.npy")
    numberOfObservations, numberOfDimensions = observationMatrix.shape
    hasObservation    = AHCFunctions.loadAvailabilityMatrix(observationMatrix)
    observationNames  = numpy.arange(numberOfObservations)


    # Generate the directory for the figures, if it does not exist yet.
    if (not os.path.exists(directoryFigures + dataSetName + "/")):
        os.makedirs(directoryFigures + dataSetName + "/")


    # Plots the observation matrix.
    if (plotObservationMatrix):
        # As not all observation matrices have 'numpy.float64' entries, we create a copy of the observation matrix for which this is true.
        observationMatrixPlot = observationMatrix.astype(numpy.float64)


        pyplot.figure(figsize = (numberOfDimensions * .3, numberOfObservations * .3))
        pyplot.imshow(observationMatrixPlot, cmap = colourMap, aspect = "auto")
        axesMain              = pyplot.gca()
        axesColourBar         = make_axes_locatable(axesMain).append_axes("right", size = colourBarWidth, pad = colourBarDistance)
        '''
        boundaries            = [0, 1, 2, 3]
        colours               = colourMap(numpy.linspace(0, 1, endpoint = True, num = 4))
        colourBarMap          = colors.ListedColormap(colours)
        colourBar             = colorbar.ColorbarBase(axesColourBar, cmap = colourBarMap, orientation = "vertical", ticks = [0.125, 0.375, 0.625, 0.875])
        colourBar.set_ticklabels(boundaries)
        '''
        colourBar             = colorbar.ColorbarBase(axesColourBar, cmap = colourMap, orientation = "vertical")

        axesMain.set_xticks(numpy.arange(numberOfDimensions))
        axesMain.set_xticklabels(dimensionsUsed, rotation = 90)
        axesMain.set_yticks(numpy.arange(numberOfObservations))
        axesMain.set_yticklabels(observationNames)
        axesMain.set_ylim(-0.5, numberOfObservations - 0.5) # use margins of 0.5 to ensure the first and last row are shown fully
        axesMain.invert_yaxis()
        axesMain.set_title(projectName + "\nobservation matrix")
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + "ObservationMatrix" + ".pdf")
        pyplot.close()


    # Plots the availability matrix.
    if (plotAvailabilityMatrix):
        pyplot.figure(figsize = (numberOfDimensions * .3, numberOfObservations * .3))
        pyplot.imshow(hasObservation, cmap = colourMapBinary, aspect = "auto")
        axesMain      = pyplot.gca()
        axesColourBar = make_axes_locatable(axesMain).append_axes("right", size = colourBarWidth, pad = colourBarDistance)
        colourBar     = colorbar.ColorbarBase(axesColourBar, cmap = colourMapBinary, orientation = "vertical")

        axesMain.set_xticks(numpy.arange(numberOfDimensions))
        axesMain.set_xticklabels(dimensionsUsed, rotation = 90)
        axesMain.set_yticks(numpy.arange(numberOfObservations))
        axesMain.set_yticklabels(observationNames)
        axesMain.set_ylim(-0.5, numberOfObservations - 0.5) # use margins of 0.5 to ensure the first and last row are shown fully
        axesMain.invert_yaxis()
        axesMain.set_title(projectName + "\nobservation availability matrix")
        pyplot.tight_layout()
        pyplot.savefig(directoryFigures + dataSetName + "/" + "AHC" + "ObservationAvailabilityMatrix" + ".pdf")
        pyplot.close()
