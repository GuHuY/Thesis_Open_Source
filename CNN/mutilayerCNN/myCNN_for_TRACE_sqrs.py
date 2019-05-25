import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
# np.random.shuffle(arr)
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)
#666 87
#73  87
LR = 0.0001  # learning rate

# 输入数据处理
# file_name = '/Users/rex/python/z_thesis/fully_combine.txt'
file_name = '/Users/rex/python/z_thesis/RR_trace_wavedet/trace_sqrs_combine.txt'

# test_data_size = 1000
# file_name = '/Users/rex/python/z_thesis/RR_data/a01.txt'

data = np.loadtxt(file_name)
nrow, nfeatures = data.shape
# nrow *= 0.1
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

tf_x = tf.placeholder(tf.float32, [None, 1*60])
ecg = tf.reshape(tf_x, [-1, 60, 1]) #  (batch, length, channel)
tf_y = tf.placeholder(tf.int32, [None, 2])# input y

# input(length, deepth): (360, 1)
# 卷积第一层
conv1 = tf.keras.layers.Conv1D(16, 3, padding='same', activation=tf.nn.relu)(ecg) # -> (360, 16)
# pool1 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv1) # -> (180, 16)

# img_shape = [128, 32, 32, 64]
# Wx_plus_b = tf.Variable(tf.random_normal(img_shape))
# axis = list(range(len(img_shape) - 1))
# wb_mean, wb_var = tf.nn.moments(Wx_plus_b, axis)


# 卷积第二层
# conv2 = tf.keras.layers.Conv1D(32, 3, padding='same', activation=tf.nn.relu)(conv1) # -> (180, 32)
# pool2 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv2) # -> (90, 32)

# 卷积第二层
# conv3 = tf.keras.layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu)(conv2) # -> (90, 64)
# pool3 = tf.keras.layers.MaxPool1D(strides=2, padding='same')(conv3) # -> (45, 64)

# 卷积输出展平
flat = tf.reshape(conv1, [-1, 60*16]) # -> (batch, 2280)         
# den = tf.keras.layers.Dense(100)(flat)            # output layer
dense1 = tf.keras.layers.Dense(60, activation=tf.nn.relu, use_bias=True,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))(flat)     
dense2 = tf.keras.layers.Dense(45, activation=tf.nn.relu, use_bias=True,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
                               )(dense1)
dense3 = tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
                               )(dense2)  
dense4 = tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
                               )(dense3)  
dense5 = tf.keras.layers.Dense(10, activation=tf.nn.relu, use_bias=True,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
                               )(dense4)                                 
## fc1 layer ##
# W_fc1 = weight_variable([45*64, 100])
# b_fc1 = bias_variable([100])
# h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(dense5) 

# 第四层，输入1024维，输出10维，也就是具体的0~9分类
# W_fc2 = weight_variable([100, 2])
# b_fc2 = bias_variable([2])
# output = tf.nn.softmax(tf.matmul(den, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
# y_ = tf.placeholder(tf.float32, [None, 2])

# 误差函数和优化模式
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
# tf.train.MomentumOptimizer(0.1, 0.9)

# _, acc_op = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, FN_op = tf.metrics.false_negatives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, FP_op = tf.metrics.false_positives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, TN_op = tf.metrics.true_negatives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, TP_op = tf.metrics.true_positives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

print('\tEpoch', '\t|    train loss\t' , '|    acc\t', '|    sen\t', '|    spe')

for step in range(10000000):
    _, loss_ = sess.run([train_op, loss], {tf_x: training_x, tf_y: training_y})
    if step % 50 == 0:
        # accuracy_1 = sess.run(acc_op, {tf_x: test_x, tf_y: test_y})
        FN = sess.run(FN_op, {tf_x: test_x, tf_y: test_y})
        FP = sess.run(FP_op, {tf_x: test_x, tf_y: test_y})
        TN = sess.run(TN_op, {tf_x: test_x, tf_y: test_y})
        TP = sess.run(TP_op, {tf_x: test_x, tf_y: test_y})
        my_acc = (TP+TN)/(FP+FN+TP+TN)
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        print('\t', step, '\t|    %.6f \t' % loss_, '|    %.6f\t' % my_acc, '|    %.6f\t' % sensitivity, '|    %.6f' % specificity )