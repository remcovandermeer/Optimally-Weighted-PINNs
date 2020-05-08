import tensorflow as tf
from PDEs.PDE import PDENeuralNetwork


# Base class for the Laplace equation, defining the PDE operators but not the domain or the boundary conditions
class LaplaceBase(PDENeuralNetwork):
    def __init__(self, domain, network=None):
        PDENeuralNetwork.__init__(self, domain, network)

        # Regular losses
        lossInt, magnitudeInt, lossBound, magnitudeBound = \
            self.ComputeLossTerms(domain, self.network.xInt, self.network.yInt, self.network.yBound,
                                  self.network.boundaryCondition)

        self.defaultLoss = tf.add(lossInt, lossBound)
        self.optimalLoss = (self.lossWeight * self.interiorDomainSize) * lossInt + \
                           ((1 - self.lossWeight) * self.totalBoundaryDomainSize) * lossBound
        self.magnitudeLoss = lossInt / magnitudeInt + lossBound / magnitudeBound

        # Validation losses
        lossIntValidate, magnitudeIntValidate, lossBoundValidate, magnitudeBoundValidate = \
            self.ComputeLossTerms(self.domain, self.network.xIntValidate, self.network.yIntValidate,
                                  self.network.yBoundValidate, self.network.boundaryConditionValidate)

        self.defaultLossValidate = tf.add(lossInt, lossBound)
        self.optimalLossValidate = (self.lossWeight * self.interiorDomainSize) * lossInt + \
                                   ((1 - self.lossWeight) * self.totalBoundaryDomainSize) * lossBound
        self.magnitudeLossValidate = lossInt / magnitudeInt + lossBound / magnitudeBound

        # Create fetch lists
        self.fetchList = [lossInt, lossBound]
        self.fetchListValidate = [lossInt, lossBound, lossIntValidate, lossBoundValidate]

    @staticmethod
    def ComputeLossTerms(domain, xInt, yInt, yBound, boundaryCondition):
        gradients = [PDENeuralNetwork.PartialDerivative(yInt, xInt[i], 2)
                     for i in range(len(domain))]
        lossInt = tf.reduce_mean(tf.square(tf.add_n(gradients)))
        magnitudeInt = tf.reduce_mean(tf.square(tf.add_n([tf.abs(grads) for grads in gradients])))
        lossBound = tf.reduce_mean(tf.square(yBound - boundaryCondition))
        magnitudeBound = tf.reduce_mean(tf.square(boundaryCondition))

        return lossInt, magnitudeInt, lossBound, magnitudeBound
