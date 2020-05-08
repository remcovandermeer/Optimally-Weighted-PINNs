import tensorflow as tf
import numpy as np
import time
from Enums import TrainMode


# Base neural network PDE solver class
class PDENeuralNetwork:
    def __init__(self, domain=None, network=None):
        self.lossWeight = tf.placeholder(tf.float64, name="lossWeight")

        self.domain = domain.copy()
        self.network = network

        # Compute domain sizes
        self.interiorDomainSize = 1
        for axis in domain:
            self.interiorDomainSize = self.interiorDomainSize * (axis[1] - axis[0])

        self.boundaryDomainSize = []
        self.totalBoundaryDomainSize = 0
        for i in range(len(domain)):
            self.boundaryDomainSize.append(1)
            for j in range(len(domain)):
                if j != i:
                    self.boundaryDomainSize[i] = self.boundaryDomainSize[i] * (self.domain[j][1] - self.domain[j][0])
            self.totalBoundaryDomainSize = self.totalBoundaryDomainSize + 2 * self.boundaryDomainSize[i]

        # Initialize loss variables
        self.defaultLoss = None
        self.optimalLoss = None
        self.magnitudeLoss = None

        self.defaultLossValidate = None
        self.optimalLossValidate = None
        self.magnitudeLossValidate = None

        self.fetchList = None
        self.fetchListValidate = None
        self.iteration = 0
        self.bestLoss = np.infty
        self.bestWeights = []
        self.bestBiases = []
        self.startTime = 0

        self.analyticalInterior = None
        self.analyticalBoundary = None
        self.analyticalInteriorMagnitude = None
        self.analyticalBoundaryMagnitude = None

    @staticmethod
    def PartialDerivative(tensor, variable, order=1):
        for i in range(order):
            if tensor is not None:
                tensor = tf.gradients(tensor, variable)[0]
        return tensor

    # region Collocation Points
    # Create feed dict with uniformly sampled data
    def SampleData(self, interiorPointCount, boundaryPointCount, validate=False, lossWeight=None):
        feedDict = dict()

        xInt = self.SampleInteriorX(interiorPointCount)
        for i in range(len(self.domain)):
            feedDict[self.network.xInt[i]] = xInt[i]

        xBound = self.SampleBoundaryX(boundaryPointCount)
        for i in range(len(self.domain)):
            feedDict[self.network.xBound[i]] = xBound[i]

        boundaryCondition = self.BoundaryCondition(xBound)
        if boundaryCondition is not None:
            feedDict[self.network.boundaryCondition] = boundaryCondition

        if validate:
            xInt = self.SampleInteriorX(interiorPointCount)
            for i in range(len(self.domain)):
                feedDict[self.network.xIntValidate[i]] = xInt[i]

            xBound = self.SampleBoundaryX(boundaryPointCount)
            for i in range(len(self.domain)):
                feedDict[self.network.xBoundValidate[i]] = xBound[i]

            boundaryCondition = self.BoundaryCondition(xBound)
            if boundaryCondition is not None:
                feedDict[self.network.boundaryConditionValidate] = boundaryCondition

        if lossWeight is not None:
            feedDict[self.lossWeight] = lossWeight

        return feedDict

    # Sample uniform collocation points in the interior of the domain
    def SampleInteriorX(self, pointCount):
        if pointCount < 1:
            pointCount = 1

        xInt = []
        for i in range(len(self.domain)):
            xInt.append(np.random.uniform(self.domain[i][0], self.domain[i][1], (pointCount, 1)))
        return xInt

    # Sample uniform collocation points on the boundary of the domain
    def SampleBoundaryX(self, pointCount):
        if pointCount < 2 * len(self.domain):
            pointCount = 2 * len(self.domain)

        xBound = []
        # Iterate over dimensions
        for i in range(len(self.domain)):
            xBound.append(np.empty((0, 1), dtype=np.float64))

            # Iterate over boundaries
            for j in range(len(self.domain)):
                for bound in self.domain[j]:
                    newPoints = max(int(pointCount * self.boundaryDomainSize[j] / self.totalBoundaryDomainSize), 1)
                    if j == i:
                        newX = np.full((newPoints, 1), bound, dtype=np.float64)
                    else:
                        newX = np.random.uniform(self.domain[j][0], self.domain[j][1],
                                                 (newPoints, 1))
                    xBound[i] = np.concatenate((xBound[i], newX))

        return xBound
    # endregion

    # region Training
    def Train(self, trainMode, interiorPointCount=2, boundaryPointCount=2, iterations=10000, lossWeight=None,
              customCallback=None):
        self.startTime = time.time()
        if trainMode == TrainMode.Default:
            self.__TrainRegular(lossFunction=self.defaultLoss,
                                interiorPointCount=interiorPointCount,
                                boundaryPointCount=boundaryPointCount,
                                iterations=iterations,
                                fetchList=self.fetchList,
                                customCallback=customCallback)

        elif trainMode == TrainMode.Optimal:
            if lossWeight is None:
                lossWeight = self.ApproximateLossWeight()
            self.__TrainRegular(lossFunction=self.optimalLoss,
                                interiorPointCount=interiorPointCount,
                                boundaryPointCount=boundaryPointCount,
                                iterations=iterations,
                                lossWeight=lossWeight,
                                fetchList=self.fetchList,
                                customCallback=customCallback)

        elif trainMode == TrainMode.Magnitude:
            self.__TrainRegular(lossFunction=self.magnitudeLoss,
                                interiorPointCount=interiorPointCount,
                                boundaryPointCount=boundaryPointCount,
                                iterations=iterations,
                                fetchList=self.fetchList,
                                customCallback=customCallback)

        elif trainMode == TrainMode.DefaultAdaptive:
            self.__TrainValidate(lossFunction=self.defaultLoss,
                                 interiorPointCount=interiorPointCount,
                                 boundaryPointCount=boundaryPointCount,
                                 iterations=iterations,
                                 fetchList=self.fetchListValidate,
                                 customCallback=customCallback)

        elif trainMode == TrainMode.OptimalAdaptive:
            if lossWeight is None:
                lossWeight = self.ApproximateLossWeight()
            self.__TrainValidate(lossFunction=self.optimalLoss,
                                 interiorPointCount=interiorPointCount,
                                 boundaryPointCount=boundaryPointCount,
                                 iterations=iterations,
                                 lossWeight=lossWeight,
                                 fetchList=self.fetchListValidate,
                                 customCallback=customCallback)

        elif trainMode == TrainMode.MagnitudeAdaptive:
            self.__TrainValidate(lossFunction=self.magnitudeLoss,
                                 interiorPointCount=interiorPointCount,
                                 boundaryPointCount=boundaryPointCount,
                                 iterations=iterations,
                                 fetchList=self.fetchListValidate,
                                 customCallback=customCallback)

    def __TrainRegular(self, lossFunction, interiorPointCount, boundaryPointCount, iterations, lossWeight=None,
                       fetchList=None, customCallback=None):
        if customCallback is None:
            callback = self.DefaultCallback
        else:
            callback = customCallback

        feedDict = self.SampleData(interiorPointCount=interiorPointCount,
                                   boundaryPointCount=boundaryPointCount,
                                   validate=False,
                                   lossWeight=lossWeight)

        self.network.Train(lossFunction, iterations, feedDict, fetchList, callback)

    def __TrainValidate(self, lossFunction, interiorPointCount, boundaryPointCount, iterations, lossWeight=None,
                        fetchList=None, customCallback=None):
        if customCallback is None:
            callback = self.DefaultCallbackValidate
            fetchList = fetchList.copy()
            fetchList.insert(0, lossFunction)
            for i in range(len(self.network.weights)):
                fetchList.append(self.network.weights[i])

            for i in range(len(self.network.biases)):
                fetchList.append(self.network.biases[i])

        else:
            callback = customCallback

        # Stop the algorithm if the collocation point counts exceed 200,000 points
        while interiorPointCount < 200000 and boundaryPointCount < 200000 and self.iteration < iterations:
            try:
                feedDict = self.SampleData(interiorPointCount=interiorPointCount,
                                           boundaryPointCount=boundaryPointCount,
                                           validate=True,
                                           lossWeight=lossWeight)

                self.bestLoss = np.infty
                self.network.Train(lossFunction, iterations - self.iteration, feedDict, fetchList, callback)
                break

            except OverFitError as e:
                # Raise the number of points
                if e.raiseInt:
                    interiorPointCount *= 2
                    print("Interior point count raised to ", interiorPointCount)
                if e.raiseBound:
                    boundaryPointCount *= 2
                    print("Boundary point count raised to ", boundaryPointCount)

                if len(self.bestWeights) > 0:
                    for i in range(len(self.network.weights)):
                        self.network.weights[i].load(self.bestWeights[i], self.network.session)

                    for i in range(len(self.network.biases)):
                        self.network.biases[i].load(self.bestBiases[i], self.network.session)

    def DefaultCallback(self, lossInt, lossBound):
        self.iteration += 1
        print("Iteration: ", self.iteration,
              ": Interior loss: ", "{:.4E}".format(lossInt),
              ", Boundary loss: ", "{:.4E}".format(lossBound),
              " Time elapsed: ", "{:.2f}".format(time.time() - self.startTime), "s")

    def DefaultCallbackValidate(self, loss, lossInt, lossBound, lossIntValidate, lossBoundValidate, *args):
        self.iteration += 1
        print("Iteration: ", self.iteration,
              ": Interior loss: ", "{:.4E}".format(lossInt),
              ", Boundary loss: ", "{:.4E}".format(lossBound),
              " Time elapsed: ", "{:.2f}".format(time.time() - self.startTime), "s")

        if loss < self.bestLoss:
            self.bestLoss = loss
            self.StoreWeights(*args)

        if lossInt < lossIntValidate / 5 or lossBound < lossBoundValidate / 5:
            raise OverFitError(lossInt < lossIntValidate / 5, lossBound < lossBoundValidate / 5)

    def StoreWeights(self, *args):
        self.bestWeights = []
        self.bestBiases = []

        counter = 0
        for i in range(len(self.network.weights)):
            self.bestWeights.append(args[counter].copy())
            counter += 1

        for i in range(len(self.network.biases)):
            self.bestBiases.append(args[counter].copy())
            counter += 1
    # endregion

    # region Plots
    # Wrapper function to get interior plot data
    def GetInteriorPlotData(self, pointCount, tensor, x):
        axisCount = 0
        for axis in x:
            if isinstance(axis, tuple):
                axisCount += 1

        if axisCount == 1:
            return self.GetInteriorPlotData1d(pointCount, tensor, x)

        elif axisCount == 2:
            return self.GetInteriorPlotData2d(pointCount, tensor, x)

        raise Exception("Bad plot domain: domain must be 1- or 2-dimensional")

    # Get interior plot data for 1d plots
    def GetInteriorPlotData1d(self, pointCount, tensor, x):
        xInt = self.RegularGridPoints(pointCount, x)
        feedDict = self.GetFeedDict(xInt=xInt)
        y = self.network.session.run(tensor, feedDict)

        xPlot = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                xPlot.append(xInt[i])

        return xPlot[0], y

    # Get interior plot data for 2d plots
    def GetInteriorPlotData2d(self, pointCount, tensor, x):
        pointCount = max(1, int(np.sqrt(pointCount)))

        xInt = self.RegularGridPoints(pointCount, x)
        feedDict = self.GetFeedDict(xInt=xInt)
        y = self.network.session.run(tensor, feedDict)

        xPlot = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                xPlot.append(xInt[i].reshape(pointCount, pointCount))

        return xPlot[0], xPlot[1], y.reshape(pointCount, pointCount)

    # Get a regular grid of points, formatted into column vectors
    @staticmethod
    def RegularGridPoints(pointCount, x):
        xInt = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                xInt.append(np.linspace(x[i][0], x[i][1], pointCount, dtype=np.float64))

            else:
                xInt.append(np.full((1, 1), x[i], dtype=np.float64))

        xIntMesh = np.meshgrid(*xInt)
        return [x.reshape(-1, 1) for x in xIntMesh]

    # Get uniform input points in the boundary of the domain, formatted into lists of column vectors
    def GetBoundaryPlotData(self, pointCount, tensor, x):
        # Compute overlap of x and the domain of the PDE, and store the axes of the plot
        xOverlap = []
        axes = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                axes.append(i)
                if x[i][0] > self.domain[i][1] or x[i][1] < self.domain[i][0]:
                    raise Exception("Requested domain does not contain the boundary")

                xMin = max(x[i][0], self.domain[i][0])
                xMax = min(x[i][1], self.domain[i][1])
                if xMin == xMax:
                    xOverlap.append(xMin)

                else:
                    xOverlap.append((xMin, xMax))

            else:
                if x[i] < self.domain[i][0] or x[i] > self.domain[i][1]:
                    raise Exception("Requested domain does not contain the boundary")

                else:
                    xOverlap.append(x[i])

        if len(axes) > 2 or len(axes) < 1:
            raise Exception("Bad plot domain: domain must be 1- or 2-dimensional")

        # Find which boundaries are included in the requested domain and store the corresponding domain
        boundIncluded = [-1, -1] * len(x)
        for i in range(len(xOverlap)):
            if isinstance(xOverlap[i], tuple):
                if xOverlap[i][0] == self.domain[i][0]:
                    dim = 0
                    for j in range(len(xOverlap)):
                        if j != i and isinstance(xOverlap[j], tuple):
                            dim += 1
                    boundIncluded[2 * i] = dim

                if xOverlap[i][1] == self.domain[i][1]:
                    dim = 0
                    for j in range(len(xOverlap)):
                        if j != i and isinstance(xOverlap[j], tuple):
                            dim += 1
                    boundIncluded[2 * i + 1] = dim

            else:
                if xOverlap[i] == self.domain[i][0]:
                    dim = 0
                    for j in range(len(xOverlap)):
                        if j != i and isinstance(xOverlap[j], tuple):
                            dim += 1
                    boundIncluded[2 * i] = dim

                if xOverlap[i] == self.domain[i][1]:
                    dim = 0
                    for j in range(len(xOverlap)):
                        if j != i and isinstance(xOverlap[j], tuple):
                            dim += 1
                    boundIncluded[2 * i + 1] = dim

        boundaryDimension = max(boundIncluded)
        if boundaryDimension < 0:
            raise Exception("Requested domain does not contain a boundary")

        # Get boundary plot data
        if boundaryDimension == 0:
            return self.GetBoundaryPlotData0d(tensor, xOverlap, boundIncluded, axes)

        elif boundaryDimension == 1:
            return self.GetBoundaryPlotData1d(pointCount, tensor, xOverlap, boundIncluded, axes)

        elif boundaryDimension == 2:
            return self.GetBoundaryPlotData2d(pointCount, tensor, xOverlap)

    # Get 0-d boundary plot data as 1-d arrays
    def GetBoundaryPlotData0d(self, tensor, x, boundIndcluded, axes):
        # Generate list with point coordinates
        xBound = []
        for i in range(len(x)):
            xBound.append([])

        for i in range(len(boundIndcluded)):
            if boundIndcluded[i] == 0:
                for j in range(len(x)):
                    if j == int(i/2):
                        xBound[j].append(self.domain[j][i % 2])

                    else:
                        xBound[j].append(x[j])

        for i in range(len(xBound)):
            xBound[i] = np.array(xBound[i]).reshape(-1, 1)

        feedDict = self.GetFeedDict(xBound=xBound)
        y = self.network.session.run(tensor, feedDict)
        output = [xBound[axis].reshape(-1) for axis in axes] + [y.reshape(-1)]
        return tuple(output)

    # Get 1-d boundary plot data as lists of 1-d arrays
    def GetBoundaryPlotData1d(self, pointCount, tensor, x, boundIncluded, axes):
        # Compute number of included boundaries
        boundaryCount = 0
        for bound in boundIncluded:
            if bound == 1:
                boundaryCount += 1

        # Create point arrays
        xBound = []
        for i in range(len(boundIncluded)):
            if boundIncluded[i] == 1:
                newX = x.copy()
                boundIndex = int(i / 2)
                newX[boundIndex] = self.domain[boundIndex][i % 2]
                xBound.append(self.RegularGridPoints(max(pointCount / boundaryCount, 2), newX))

        # Evaluate tensor
        y = []
        xOutput = []
        for j in range(len(axes)):
            xOutput.append([])

        for i in range(len(xBound)):
            feedDict = self.GetFeedDict(xBound=xBound[i])
            y.append(self.network.session.run(tensor, feedDict).reshape(-1))

            for j in range(len(axes)):
                xOutput[j].append(xBound[i][axes[j]].reshape(-1))

        return tuple(xOutput + [y])

    # Get 2-d boundary plot data
    def GetBoundaryPlotData2d(self, pointCount, tensor, x):
        pointCount = max(1, int(np.sqrt(pointCount)))

        xBound = self.RegularGridPoints(pointCount, x)
        feedDict = self.GetFeedDict(xBound=xBound)
        y = self.network.session.run(tensor, feedDict)

        xPlot = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                xPlot.append(xBound[i].reshape(pointCount, pointCount))

        return xPlot[0], xPlot[1], y.reshape(pointCount, pointCount)

    # Override this to define boundary conditions
    def BoundaryCondition(self, x):
        return None

    def GetFeedDict(self, xInt=None, xBound=None, xIntValidate=None, xBoundValidate=None, lossWeight=None):
        feedDict = dict()

        if xInt is not None:
            for i in range(len(self.domain)):
                feedDict[self.network.xInt[i]] = xInt[i]

        if xBound is not None:
            for i in range(len(self.domain)):
                feedDict[self.network.xBound[i]] = xBound[i]

            boundaryCondition = self.BoundaryCondition(xBound)
            if boundaryCondition is not None:
                feedDict[self.network.boundaryCondition] = boundaryCondition

        if xIntValidate is not None:
            for i in range(len(self.domain)):
                feedDict[self.network.xIntValidate[i]] = xInt[i]

        if xBoundValidate is not None:
            for i in range(len(self.domain)):
                feedDict[self.network.xBoundValidate[i]] = xBoundValidate[i]

            boundaryCondition = self.BoundaryCondition(xBoundValidate)
            if boundaryCondition is not None:
                feedDict[self.network.boundaryConditionValidate] = boundaryCondition

        if lossWeight is not None:
            feedDict[self.lossWeight] = lossWeight

        return feedDict
    # endregion

    # region Computations
    def ApproximateLossWeight(self, samplePoints=500000):
        if self.analyticalInteriorMagnitude is None or self.analyticalBoundaryMagnitude is None:
            raise NotImplemented("PDE does not define analytical magnitudes")

        feedDict = self.SampleData(interiorPointCount=samplePoints, boundaryPointCount=samplePoints)

        magInt = self.network.session.run(self.analyticalInteriorMagnitude, feedDict)
        magBound = self.network.session.run(self.analyticalBoundaryMagnitude, feedDict)
        result = magBound / (magInt + magBound)

        print("Optimal Loss Weight: ", "{:.4E}".format(result))
        return result

    def ComputeL2Error(self, relative=True, samplePoints=500000):
        if self.analyticalInterior is None:
            raise NotImplemented("PDE does not define an analytical solution")

        feedDict = self.SampleData(interiorPointCount=samplePoints, boundaryPointCount=samplePoints)
        prediction = self.network.session.run(self.network.yInt, feedDict)
        analytical = self.network.session.run(self.analyticalInterior, feedDict)

        if relative:
            error = np.sqrt(np.sum((prediction - analytical) ** 2)) / np.sqrt(np.sum(analytical ** 2))
            print("Relative L2 error: ", "{:.4E}".format(error))

        else:
            error = np.sqrt(np.sum((prediction - analytical) ** 2))
            print("L2 error: ", "{:.4E}".format(error))

    def ComputeMaxError(self, relative=True, samplePoints=500000):
        if self.analyticalInterior is None:
            raise NotImplemented("PDE does not define an analytical solution")

        feedDict = self.SampleData(interiorPointCount=samplePoints, boundaryPointCount=samplePoints)
        prediction = self.network.session.run(self.network.yInt, feedDict)
        analytical = self.network.session.run(self.analyticalInterior, feedDict)

        if relative:
            error = np.max(prediction - analytical) / np.max(analytical)
            print("Relative L_Infinity error: ", "{:.4E}".format(error))

        else:
            error = np.max(prediction - analytical)
            print("L_Infinity error: ", "{:.4E}".format(error))
    # endregion


# Exception class used to terminate the scipy optimizer interface
class OverFitError(Exception):
    def __init__(self, raiseInt, raiseBound):
        self.raiseInt = raiseInt
        self.raiseBound = raiseBound

