import numpy as np

class Node:
    def __init__(self, children = [], value = None, topdiff = 0):
        self.children = children
        self.value = value
        self.topdiff = topdiff

    def forward(self):
        for child in self.children:
            child.forward()

    def backward(self):
        pass


class AddNode(Node):
    def forward(self):
        super().forward()
        self.value = self.children[0].value + self.children[1].value

    def backward(self):
        for c in children:
            c.topdiff = self.topdiff

class MulNode(Node):
    def forward(self):
        super().forward()
        self.value = self.children[0].value * self.children[1].value

    def backward(self):
        self.children[0].topdiff = self.children[1].value * self.topdiff
        self.children[1].topdiff = self.children[0].value * self.topdiff

class ExpNode(Node):
    def forward(self):
        super().forward()
        self.value = np.exp(self.children[0].value)

    def backward(self):
        self.children[0].topdiff = np.exp(self.children[0].value)
