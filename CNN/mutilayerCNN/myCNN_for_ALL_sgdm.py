import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
# np.random.shuffle(arr)
tf.set_random_seed(1)
np.random.seed(1)
random.seed(5)
#666 87
#73  87
# LR = 0.0001  # learning rate
LR = 0.0001  # quicker train

# 输入数据处理
data_name = 'RR_PT1'
# file_path = '/Users/rex/python/Thesis_Open_Source/' + data_name + '/'
# # data_name = 'RR_PT2_wavedet'
# data_name = 'White_Noise_Test/RV_noise' 
file_path = '/Users/rex/python/Thesis_Open_Source/' + data_name + '/'

# test_data_size = 1000
# file_name = '/Users/rex/python/z_thesis/RR_data/a01.txt'

data = np.loadtxt(file_path+'combine.txt')
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

training_x = data[training_head:training_tail, :-1].astype(np.float32)
# [...0, 1, 0...] -> [...[1, 0], [0, 1], [1, 0]...]
label_transformation = [[int(not x), x] for x in data[training_head:training_tail, -1]]
training_y = np.array(label_transformation).astype(np.float32)

test_x = data[test_head:test_tail, :-1].astype(np.float32)
test_y = np.array([[int(not x), x] for x in data[test_head:test_tail, -1]]).astype(np.float32)

tf_x = tf.placeholder(tf.float32, [None, 60])
my_input = tf.reshape(tf_x, [-1, 60, 1]) 
tf_y = tf.placeholder(tf.int32, [None, 2])

# 卷积第一层
conv1 = tf.keras.layers.Conv1D(16, 3, padding='same', activation=tf.nn.relu)(my_input) # -> (60, 16)

# 卷积输出展平
flat = tf.reshape(conv1, [-1, 60*16]) # -> (960)

# 全连接层
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
# 输出层                                 
output = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(dense5) 


# 误差函数和优化模式
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) 
train_op = tf.train.MomentumOptimizer(LR, 0.9, use_nesterov=True).minimize(loss)
# tf.train.MomentumOptimizer(0.1, 0.9)
# tf.train.AdamOptimizer(LR)
_, AUC_op = tf.metrics.auc(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, FN_op = tf.metrics.false_negatives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, FP_op = tf.metrics.false_positives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, TN_op = tf.metrics.true_negatives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
_, TP_op = tf.metrics.true_positives(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))
saver = tf.train.Saver(max_to_keep=4)
tf.summary.scalar('AUC_', AUC_op)
tf.summary.scalar('Acc_', (TP_op+TN_op)/(FP_op+FN_op+TP_op+TN_op))
tf.summary.scalar('Sen_', TP_op/(TP_op+FN_op))
tf.summary.scalar('Spe_', TN_op/(FP_op+TN_op))
tf.summary.scalar('Loss_', loss)
# tf.summary.scalar('zFN', FN_op)
# tf.summary.scalar('zFP', FP_op)
# tf.summary.scalar('zTN', TN_op)
# tf.summary.scalar('zTP', TP_op)
merge_summary = tf.summary.merge_all()


sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph
train_writer = tf.summary.FileWriter(file_path+'log_sgdm/', sess.graph)
print('\tEpoch', '\t|    train loss\t' , '|    acc\t', '|    sen\t', '|    spe\t', '|    auc' )

for epoch in range(10000000):
    _, loss_ = sess.run([train_op, loss], {tf_x: training_x, tf_y: training_y})
    if epoch % 50 == 0:
        FN, FP, TN, TP, AUC, train_summary = sess.run([FN_op, FP_op, TN_op, TP_op, AUC_op, merge_summary],
                                                       {tf_x: test_x, tf_y: test_y})
        accuracy = (TP+TN)/(FP+FN+TP+TN)
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        train_writer.add_summary(train_summary, epoch)
        saver.save(sess, file_path+'model_sgdm/', global_step=epoch)
        print('\t', epoch, '\t|    %.6f \t' % loss_, '|    %.6f\t' % accuracy, '|    %.6f\t' % sensitivity, '|    %.6f' % specificity, '|    %.6f' % AUC )
train_writer.close()