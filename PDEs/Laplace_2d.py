import tensorflow as tf
import numpy as np
from PDEs.LaplaceBase import LaplaceBase


class Laplace_2d(LaplaceBase):
    def __init__(self, frequency, network=None):
        domain = [(0, 1), (0, 1)]
        self.frequency = frequency
        LaplaceBase.__init__(self, domain, network)

        # Analytical Solution
        self.analyticalInterior = self.AnalyticalSolution(self.network.xInt)
        self.analyticalBoundary = self.AnalyticalSolution(self.network.xBound)

        # Analytical magnitudes
        gradients = [self.PartialDerivative(self.analyticalInterior, self.network.xInt[i], 2)
                     for i in range(len(domain))]
        self.analyticalInteriorMagnitude = self.interiorDomainSize * \
            tf.reduce_mean(tf.square(tf.add_n([tf.abs(grads) for grads in gradients])))
        self.analyticalBoundaryMagnitude = self.totalBoundaryDomainSize * \
            tf.reduce_mean(tf.square(self.network.boundaryCondition))

    def AnalyticalSolution(self, x):
        return tf.exp(-x[0] * self.frequency) * tf.sin(x[1] * self.frequency)

    def BoundaryCondition(self, x):
        return np.exp(-x[0] * self.frequency) * np.sin(x[1] * self.frequency)
