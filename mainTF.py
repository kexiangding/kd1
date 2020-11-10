# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
import tensorflow as tf
version = tf.__version__
gpu_ok = tf.test.is_gpu_available()
print("tf version:",version,"\nuse GPU",gpu_ok)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 添加神经网络层
def add_layer(inputs,in_size,out_size,activation_function=None):
   Weights = tf.Variable(tf.random_normal([in_size,out_size]))
   biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs,Weights) + biases

   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs

# 构建
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys =tf.placeholder(tf.float32,[None,1])

# 建立输入层 隐藏层 输出层
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 预测
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

sess = tf.Session()
sess.run(init)
for i in range(1000):
 training
   sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
   if i % 20 == 0:
       # to visualize the result and improvement
       try:
           ax.lines.remove(lines[0])
       except Exception:
           pass
       prediction_value = sess.run(prediction, feed_dict={xs: x_data})
       # plot the prediction
       lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
       plt.pause(0.01)

print('over.')
sess.close()