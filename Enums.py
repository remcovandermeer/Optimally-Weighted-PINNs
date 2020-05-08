from enum import Enum


class ActivationFunction(Enum):
    Tanh = 0
    Sigmoid = 1
    Sin = 2
    Cos = 3
    Atan = 4


class TrainMode(Enum):
    Default = 0
    Optimal = 1
    Magnitude = 2
    DefaultAdaptive = 3
    OptimalAdaptive = 4
    MagnitudeAdaptive = 5
