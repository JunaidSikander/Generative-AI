import numpy as np

# Create two vectors (arrays)
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# Dot product of two vectors
dot_product = np.dot(vector_a, vector_b)

print("Dot product:", dot_product)

# Common NumPy functionalities (examples)
print("Zeros array:", np.zeros(3))
print("Ones array:", np.ones(3))
print("Range array:", np.arange(1, 10, 2))
print("Mean of vector_a:", np.mean(vector_a))
print("Random number:", np.random.rand())