import tensorflow as tf
import numpy as np
import os
# path = os.getcwd()
# file_name = path+'\\fully_combine.txt'

# np.random.shuffle(arr)
tf.set_random_seed(1)
np.random.seed(1)

#
LR = 0.0001  # learning rate


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 输入数据处理
file_name = '/Users/rex/python/z_thesis/fully_combine.txt'
# file_name = '/Users/rex/python/z_thesis/RR_data/a01.txt'
data = np.loadtxt(file_name)
#13660
training_x = data[1000:, :-1].astype(np.float32)
# [...0, 1, 0...] -> [...[1, 0], [0, 1], [1, 0]...]
label_transformation = [[int(not x), x] for x in data[1000:, -1]]
training_y = np.array(label_transformation).astype(np.float32)
test_x = data[0:1000, :-1].astype(np.float32)
test_y = np.array([[int(not x), x] for x in data[0:1000, -1]]).astype(np.float32)

tf_x = tf.placeholder(tf.float32, [None, 1*360])
ecg = tf.reshape(tf_x, [-1, 360, 1]) #  (batch, length, channel)
tf_y = tf.placeholder(tf.int32, [None, 2])# input y

# input(length, deepth): (360, 1) 
# 卷积第一层
conv1 = tf.keras.layers.Conv1D(16, 5, padding='same', activation=tf.nn.relu)(ecg) # -> (360, 16)
pool1 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv1) # -> (180, 16)

# 卷积第二层
conv2 = tf.keras.layers.Conv1D(32, 5, padding='same', activation=tf.nn.relu)(pool1) # -> (180, 32)
pool2 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv2) # -> (90, 32)

# 卷积第二层
conv3 = tf.keras.layers.Conv1D(64, 5, padding='same', activation=tf.nn.relu)(pool2) # -> (90, 64)
pool3 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv3) # -> (45, 64)

# 卷积输出展平
flat = tf.reshape(pool3, [-1, 45*64]) # -> (batch, 2280)         
# den = tf.keras.layers.Dense(100)(flat)            # output layer
dense1 = tf.keras.layers.Dense(20, activation=tf.nn.relu,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))(flat)     

## fc1 layer ##
# W_fc1 = weight_variable([45*64, 100])
# b_fc1 = bias_variable([100])
# h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

output = tf.keras.layers.Dense(2)(dense1) 

# 第四层，输入1024维，输出10维，也就是具体的0~9分类
# W_fc2 = weight_variable([100, 2])
# b_fc2 = bias_variable([2])
# output = tf.nn.softmax(tf.matmul(den, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
# y_ = tf.placeholder(tf.float32, [None, 2])

# 误差函数和优化模式
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

print('\tStep', '\t| train loss\t' , '| test accuracy')

for step in range(2000):
    _, loss_ = sess.run([train_op, loss], {tf_x: training_x, tf_y: training_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print('\t', step, '\t|    %.4f \t' % loss_, '|    %.2f' % accuracy_)