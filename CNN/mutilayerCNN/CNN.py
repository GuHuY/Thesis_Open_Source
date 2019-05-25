import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
#
BATCH_SIZE = 50
LR = 0.00   # learning rate


mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]


# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0])); plt.show()

tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

# 卷积第一层
# shape (28, 28, 1)
conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', 
                               activation=tf.nn.relu)(image)         # -> (28, 28, 16)
pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv1)           # -> (14, 14, 16)

# 卷积第二层
conv2 = tf.keras.layers.Conv2D(32, 5, 1, 'same', activation=tf.nn.relu)(pool1)    # -> (14, 14, 32)
pool2 = tf.keras.layers.MaxPool2D(2, 2)(conv2)   # -> (7, 7, 32)

# 卷积输出
flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
output = tf.keras.layers.Dense(10)(flat)             # output layer

# 误差函数和优化模式
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

for step in range(2000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)