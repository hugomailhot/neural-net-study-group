# How to load data from a pickle file in Python

```python
with open('filename.pickle', 'rb') as f:
    data = pickle.load(f)
```

Whatever was in the pickle file is now in the `data` variable.
You must open the file using in read binary mode, hence the `'rb'` argument
in the `open()` function call.


# Description of the datasets

Please note that all pickled datasets where serialized using the most advanced pickle protocol, 
which is only available in Python 3. In short, use Python 3 to unpickle them.

### moons.pickle

Two sets of points in a 2-d space, one with where the label is `0`, the other where the label is `1`.
The sets form two interlocked crescents, making them non linearly separable. This forces
the use of a classifier that can express a non linear decision boundary, such as a neural network
with non-linear units.

### scatter.pickle

A set of points in 2-d with high correlation between _x_ and _y_ coordinates, the perfect dataset for
a linear regression using a polynomial of degree 1.
