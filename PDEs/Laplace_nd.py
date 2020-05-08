import tensorflow as tf
import numpy as np
from PDEs.LaplaceBase import LaplaceBase


class Laplace_nd(LaplaceBase):
    def __init__(self, frequencies, network=None):
        domain = [(0, 1)] * len(frequencies)
        self.frequencies = frequencies
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
        y = tf.exp(-x[0] * self.frequencies[0])
        for i in range(1, len(self.frequencies)):
            y = y * tf.sin(x[i] * self.frequencies[i])
        return y

    def BoundaryCondition(self, x):
        y = np.exp(-x[0] * self.frequencies[0])
        for i in range(1, len(self.frequencies)):
            y = y * np.sin(x[i] * self.frequencies[i])
        return y
