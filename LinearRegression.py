import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


num_points = 1000
target_vectors_set = []
data_vectors_set = []

learning_rate = 0.25
iteration_stop_condition = 0.0008
num_iterations = 64

np.random.seed(0)

for i in range(num_points):
    xt = np.random.normal(0.0, 0.55)
    yt = xt * 0.1 + 0.3
    yd = yt + np.random.normal(0.0, 0.03)
    target_vectors_set.append([xt, yt])
    data_vectors_set.append([xt, yd])

x_target = [v[0] for v in target_vectors_set]
y_target = [v[1] for v in target_vectors_set]

x_data = [v[0] for v in data_vectors_set]
y_data = [v[1] for v in data_vectors_set]


# Graph (Linear Regression)
plt.figure()
targetPlt = plt.subplot()
dataPlt = plt.subplot()
resultPlt = plt.subplot()
lossPlt = plt.subplot()
targetPlt.plot(x_target, y_target, 'k', label='Target (y=0.1x+0.3)')
plt.suptitle('Dataset', fontsize=14)
plt.xlabel('x')
plt.xlim(-2, 2)
plt.ylim(0.0, 0.65)
plt.ylabel('y')
plt.legend(loc='upper left')
plt.draw()

plt.pause(2)

dataPlt.plot(x_data, y_data, 'ro', label='Random Samples')
plt.legend(loc='upper left')
plt.draw()

plt.pause(2)

targetPlt.plot(x_target, y_target, 'k')
plt.suptitle('Linear Regression', fontsize=14)
plt.draw()

plt.pause(2)


# Linear Regression using tensorflow
tf.set_random_seed(0)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
#W = tf.Variable(tf.random_normal([1], 0.0, 1.732050))
#W = tf.Variable(tf.truncated_normal([1], 0.0, 1.732050))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

y_train = tf.Variable(tf.zeros([1]))
l_train = tf.Variable(tf.zeros([1]))


optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

resultPlt.plot(x_data, sess.run(y), 'm-')

step = tf.Variable(tf.zeros([1]))
for step in range(num_iterations):
    sess.run(train)
    y_train = sess.run(y)
    l_train = sess.run(loss)
    print(step, sess.run(W), sess.run(b), l_train)
    if step < num_iterations:
        resultPlt.plot(x_data, y_train, 'y')
        lossPlt.plot(step/num_iterations, l_train, 'co')
        if step == 0:
            plt.text(0.0, l_train + 0.02, '%s'+str(step+1), text=l_train)
            plt.text(0.0, l_train + 0.04, 'Loss:')
        plt.draw()
        plt.pause(0.005)
    if l_train < iteration_stop_condition:
        break


# Graph (Result)
resultPlt.plot(x_data, y_train, 'b', label='Linear Regression')
lossPlt.plot(step/num_iterations, l_train, 'co', label='Loss')
plt.text(step/num_iterations, 0.02, '%s', text=l_train)
plt.text(step/num_iterations, 0.06, '%s', text=learning_rate)
plt.text(step/num_iterations, 0.04, 'Loss:')
plt.text(step/num_iterations, 0.08, 'LR:')
plt.legend(loc='upper left')
plt.draw()

plt.pause(1)

plt.figure()
plt.plot(x_target, y_target, 'r', label='Target (y=0.1x+0.3)')
plt.plot(x_data, y_train, 'b', label='Linear Regression')
plt.text(0.0, 0.1, '%s', text=1.0-l_train, fontsize=14)
plt.text(0.0, 0.13, 'Accuracy:', fontsize=14)
plt.suptitle('Result', fontsize=14)
plt.xlabel('x')
plt.xlim(-2, 2)
plt.ylim(0.0, 0.65)
plt.ylabel('y')
plt.legend(loc='upper left')
plt.draw()

plt.show(block=True)
