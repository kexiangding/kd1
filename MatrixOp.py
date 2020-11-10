
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

class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
#dictionary
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
print(d)
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"

#set
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"

#from math import sqrt
import math
#nums = {tf.sqrt(x) for x in range(10)}
nums = {int(np.sqrt(x)) for x in range(10)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"

#tuple
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
print(d)

#SciPy
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('C:\KD\KenDc\S\hs\old\Cal12\jie\pic\P1000728.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('C:\KD\KenDc\S\hs\old\Cal12\jie\pic\P1.jpg', img_tinted)
print(img_tinted.dtype, img_tinted.shape)

#matlab
import numpy as np
import matplotlib.pyplot as plt
# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

#Solve equations of multiple variable regression多元回归
import csv
#import numpy as np

def readData():
    X = []
    y = []
    with open('C:\KD\KenDc\kdAemTx\Python\Housing.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        # Read X and y
        for line in rdr:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
            X.append(xline)
            y.append(float(line[-1]))
    return (X,y)

X0,y0 = readData()
# Convert all but the last 10 rows of the raw data to numpy arrays
d = len(X0)-10
X = np.array(X0[:d])
y = np.transpose(np.array([y0[:d]]))

# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)
print(beta)

# Make predictions for the last 10 rows in the data set
for data,actual in zip(X0[d:],y0[d:]):
    x = np.array([data])
    prediction = np.dot(x,beta)
    print('prediction = '+str(prediction[0,0])+' actual = '+str(actual))