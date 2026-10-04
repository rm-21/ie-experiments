import numpy as np

x = np.ones((2, 3, 4))
w = np.ones((1, 4, 6))

y = x @ w
print(y)
assert y.shape == (2, 3, 6)
assert np.all(y == 4)

for shape in [(2, 5, 6), (3, 4, 6)]:
    try:
        x @ np.ones(shape)
    except ValueError:
        print("Rejected: ", shape)
    else:
        raise AssertionError("Expected a shape error")