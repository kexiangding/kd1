
import tensorflow as tf
import numpy as np

#x = tf.ones((2, 2), dtype=tf.dtypes.float32)
#y = tf.constant([[1, 2],[3, 4]], dtype=tf.dtypes.float32)
#z = tf.matmul(x, y)
x = np.ones((3, 2), dtype=np.float)
y = np.array([[1, 2], [3, 4]], dtype=np.float)
z = x@y#np.matmul(x, y)
#print(x)
#print(y)
print(z)
# tf.Tensor(
# [[4. 6.]
#  [4. 6.]], shape=(2, 2), dtype=float32)
#znp=z.numpy()
#print(z.numpy())
#print(znp)
# [[4. 6.]
# [4. 6.]]
print(z.T.shape)

print(y is x)
w=x.copy()
print(w)
palette = np.array( [ [0,0,0],                # black
                       [255,0,0],              # red
                       [0,255,0],              # green
                       [0,0,255],              # blue
                       [255,255,255] ] )       # white
image = np.array( [ [ 0, 1, 2, 0 ],           # each value corresponds to a color in the palette
                     [ 0, 3, 4, 0 ]  ] )
print(palette[image])

import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 1#0.5
v = np.random.normal(mu,sigma,10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=1)       # matplotlib version (plot)
plt.show()
# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5*(bins[1:]+bins[:-1]), n)
plt.show()