import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from NeuralNetwork import NeuralNetwork
from PDEs.Laplace_nd import Laplace_nd
from Enums import ActivationFunction, TrainMode


def main():
    # Create Neural Network
    network = NeuralNetwork(inputDimension=3, hiddenLayers=[20, 20, 20, 20], activationFunction=ActivationFunction.Tanh)

    # Load weights
    # network.LoadWeights(path)

    # Create PDE
    frequencies = [0, 4 * np.pi, 1 * np.pi]
    frequencies[0] = np.sqrt(np.sum([freq ** 2 for freq in frequencies]))
    laplace = Laplace_nd(frequencies=frequencies, network=network)

    # Train
    # laplace.Train(trainMode=TrainMode.DefaultAdaptive, iterations=20000)
    # laplace.Train(trainMode=TrainMode.OptimalAdaptive, iterations=20000)
    laplace.Train(trainMode=TrainMode.MagnitudeAdaptive, iterations=2000)

    # Store weights
    # network.SaveWeights(path)

    laplace.ComputeL2Error()
    laplace.ComputeMaxError()

    # region Plots
    # Plot approximation at x_1 = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = laplace.GetInteriorPlotData(pointCount=5000, tensor=network.yInt, x=[0, (0, 1), (0, 1)])
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$u$")

    # Plot analytical solution at x_1 = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = laplace.GetInteriorPlotData(pointCount=5000, tensor=laplace.analyticalInterior, x=[0, (0, 1), (0, 1)])
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$u$")

    # Plot boundary condition at x_1 = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = laplace.GetBoundaryPlotData(pointCount=5000, tensor=network.boundaryCondition, x=[0, (0, 1), (0, 1)])
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$u$")

    # Plot solution and boundary conditions at x_3 = 0.5
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = laplace.GetInteriorPlotData(pointCount=5000, tensor=network.yInt, x=[(0, 1), (0, 1), 0.5])
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$u$")

    # Boundary conditions
    color = next(ax._get_lines.prop_cycler)['color']
    x, y, z = laplace.GetBoundaryPlotData(pointCount=1000, tensor=network.boundaryCondition, x=[(0, 1), (0, 1), 0.5])
    ax.plot3D(x[0], y[0], z[0], color=color, label="Boundary Condition")
    for i in range(1, len(x)):
        ax.plot3D(x[i], y[i], z[i], color=color)

    plt.show()
    # endregion

    # Cleanup
    network.Cleanup()
