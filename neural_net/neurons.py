
class _Neuron:
    def __init__(self):
        self.value = 0


class InputNeuron(_Neuron):
    def __init__(self):
        super().__init__()

class DeepNeuron(_Neuron):
    def __init__(self, weights, bias):
        super().__init__()
        self.derivative_value = 0
        self.weights_derivatives = []
        self.weights = weights
        self.bias = bias
        self.error = 0

class OutputNeuron(_Neuron):
    def __init__(self, weights, bias):
        super().__init__()
        self.derivative_value = 0
        self.weights_derivatives = []
        self.weights = weights
        self.bias = bias
        self.error = 0
