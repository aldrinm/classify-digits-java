import numpy as np

def sigmoid_double(x):
  return 1.0 / (1.0 + np.exp(-x))

def sigmoid(z):
  return np.vectorize(sigmoid_double)(z)

def sigmoid_prime_double(x):
  return sigmoid_double(x) * (1-sigmoid_double(x))

def sigmoid_prime(z):
  return np.vectorize(sigmoid_prime_double)(z)

print(f"sigmoid_double(800) = {sigmoid_double(800)}")

# in1 = [ [1,2,3], [4,5,6]]
# out1 = sigmoid(in1)
# print(f"out1 = {out1}")

# sigmoid prime is sigmoid derivative
# in2 = [ [1,2,3], [4,5,6]]
in2 = [ 0,1,2,3, 4,5,6,7,8,9]
out2 = sigmoid_prime(in2)
print(f"out2 = {out2}")


# def loss_function(predictions, labels):
#   diff = predictions - labels
#   return 0.5 * sum(diff * diff)[0]
#
# predictions1 = [1,2,3]
# labels1 = [2,2,2]
# loss = loss_function(predictions1, labels1)
# print(f"loss = {loss}")

# dot product
# a = [1, 2, 3]
# b = [4, 5, 6]
# print(f"a.b ={np.dot(a, b)}")

w = np.random.rand(2, 4)
print(f"w = {w}")
print(f"shape = {w.shape}")