# This is a graph implementing f(x, y, z) = w1x * w2y + w3z

from node import Node, AddNode, MulNode

x = Node(value=3)
y = Node(value=4)
z = Node(value=5)

xy = MulNode(children=[x, y])
xypz = AddNode(children=[xy, z])

xypz.forward()
# Value should be 17
print(xypz.value)

