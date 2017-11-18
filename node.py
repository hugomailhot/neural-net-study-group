import numpy as np
from functools import reduce
from operator import mul

class Node:
    def __init__(self, children = [], value = None, topdiff = 0):
        self.children = children
        self.value = value
        self.topdiff = topdiff

    def forward(self):
        # Reset topdiff, preparing for accumulation in the backward pass
        self.topdiff = 0
        for child in self.children:
            child.forward()

    def backward(self):
        for child in self.children:
            child.backward()

    def accumulate_gradient(self, gradient):
        self.topdiff += gradient

class ConstantNode(Node):
    """Constant Node

    This node represents a constant input to the model.
    """
    def __init__(self, value = None, topdiff = 0):
        super().__init__([], value, topdiff)

class ParameterNode(Node):
    """Parameter Node

    This node represents a parameter in the model.
    """
    def __init__(self, value = None, topdiff = 0):
        super().__init__([], value, topdiff)

class AddNode(Node):
    """Addition Node

    This node adds its inputs.
    """
    def forward(self):
        super().forward()
        self.value = sum(child.value for child in self.children)

    def backward(self):
        for c in children:
            c.accumulate_gradient(self.topdiff)
        super().backward()


class MulNode(Node):
    """Multiplication Node

    This node multiplies its two inputs.
    """
    def forward(self):
        super().forward()
        # This is the product equivalent of sum(list)
        self.value = reduce(mul, (child.value for child in self.children))

    def backward(self):
        self.children[0].accumulate_gradient(self.children[1].value *
                                             self.topdiff)
        self.children[1].accumulate_gradient(self.children[0].value *
                                             self.topdiff)
        super().backward()


class ExpNode(Node):
    """Exponential Function Node

    This node computes the exponential function with the node's input as the
    exponent.
    """
    def forward(self):
        super().forward()
        self.value = np.exp(self.children[0].value)

    def backward(self):
        self.children[0].accumulate_gradient(np.exp(self.children[0].value))
        super().backward()


class SquaredLossNode(Node):
    def __init__(self, pred, true):
        super().__init__(children=[pred, true])

    def forward(self):
        super().forward()
        self.value = (self.predicted - self.true)**2

    def backward(self):
        pred = self.children[0]
        true = self.children[1]
        self.children[0].accumulate_gradient(2 * (pred - true) * self.topdiff)
        self.children[1].accumulate_gradient(-2 * (pred - true) * self.topdiff)
        super().backward()
