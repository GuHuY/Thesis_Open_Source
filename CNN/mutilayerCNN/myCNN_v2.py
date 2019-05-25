import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# np.random.shuffle(arr)
tf.set_random_seed(1)
np.random.seed(1)

#
LR = 0.01  # learning rate

# 输入数据处理
# file_name = '/Users/rex/python/z_thesis/fully_combine.txt'
# test_data_size = 1000
file_name = '/Users/rex/python/z_thesis/RR_trace/c01.txt'
# file_name = '/Users/rex/python/z_thesis/RR_tracefully_combine.txt'

data = np.loadtxt(file_name)#[2000:4000]
nrow, nfeatures = data.shape
random.shuffle(data)
# wedge = nrow/10
# test_head = 0
# test_tail = int(wedge/10)
# training_head = int(wedge/10)
# training_tail = int(wedge)

test_head = 0
test_tail = int(nrow/10)
training_head = int(nrow/10)
training_tail = int(nrow)
#13660

training_x = data[training_head:training_tail, :-1].astype(np.float32)
# [...0, 1, 0...] -> [...[1, 0], [0, 1], [1, 0]...]
label_transformation = [[int(not x), x] for x in data[training_head:training_tail, -1]]
training_y = np.array(label_transformation).astype(np.float32)

test_x = data[test_head:test_tail, :-1].astype(np.float32)
test_y = np.array([[int(not x), x] for x in data[test_head:test_tail, -1]]).astype(np.float32)

tf_x = tf.placeholder(tf.float32, [None, 60])
ecg = tf.reshape(tf_x, [-1, 60, 1]) #  (batch, length, channel)
tf_y = tf.placeholder(tf.int32, [None, 2])# input y


# 卷积第一层
conv1 = tf.keras.layers.Conv1D(16, 5, padding='same', activation=tf.nn.relu)(ecg) # -> (60, 16)
# pool1 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv1) # -> (15, 16)

# 卷积第二层
conv2 = tf.keras.layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu)(conv1) # -> (60, 32)
# pool2 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv2) # -> (15, 32)

# 卷积第二层
conv3 = tf.keras.layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu)(conv2) # -> (60, 64)
# pool3 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv3) # -> (, 64)

# 卷积输出展平
flat = tf.reshape(conv3, [-1, 60*64]) # -> (batch, 1920)         
# den = tf.keras.layers.Dense(100)(flat)            # output layer
dense1 = tf.keras.layers.Dense(100, activation=tf.nn.relu,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))(flat)     
dense2 = tf.keras.layers.Dense(10, activation=tf.nn.relu,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))(dense1)
dense3 = tf.keras.layers.Dense(10, activation=tf.nn.relu,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))(dense2)  

output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(dense3) 

# 误差函数和优化模式
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

_, acc_op = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
sess.run(init_op)     # initialize var in graph

print('\tStep', '\t|    train loss\t' , '| test accuracy')
for step in range(3201):
    _, loss_ = sess.run([train_op, loss], {tf_x: training_x, tf_y: training_y})
    if step % 5 == 0:
        accuracy_1 = sess.run(acc_op, {tf_x: test_x, tf_y: test_y})
        print('\t', step, '\t|    %.6f \t' % loss_, '|    %.4f' % accuracy_1)
