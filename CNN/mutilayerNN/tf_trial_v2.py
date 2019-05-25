from ty_lib import *
import random
from sklearn.model_selection import KFold



def compute_accuracy(v_data, v_lable):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_data})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_lable, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_data, ys: v_lable})
    return result * 100

file_name = '/Users/rex/python/z_thesis/RR_trace/a01.txt'

data = np.loadtxt(file_name).tolist()
random.shuffle(data)
kf = KFold(n_splits=10)

accuracy = []
for k in range(5):
    for train, test in kf.split(data):
        traning = np.array(data)[train]
        test = np.array(data)[test]
        training_data = traning[:, :-1].astype(np.float32)
        training_label = np.array([[int(not x), x] for x in traning[:, -1]]).astype(np.float32)
        vali_data = test[:, :-1].astype(np.float32)
        vali_label = np.array([[int(not x), x] for x in test[:, -1]]).astype(np.float32)


        # x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
        # noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
        # y_data = np.square(x_data) - 0.5 + noise

        xs = tf.placeholder(tf.float32, [None, 60])
        ys = tf.placeholder(tf.float32, [None, 2])

        # hidden layers
        l1 = add_layer(xs, 60, 30)
        l2 = add_layer(l1, 30, 10)
        l3 = add_layer(l2, 10, 5)
        prediction = add_layer(l3, 5, 2, tf.nn.softmax) # output layer

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                    reduction_indices=[1])) 
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        # GradientDescentOptimizer(0.7).minimize(cross_entropy)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            # training
            sess.run(train_step, feed_dict={xs: training_data, ys: training_label})
        accuracy.append(compute_accuracy(vali_data, vali_label))
        # print("fold "+str(len(accuracy))+"   ACC "+ "%.2f%%" % float(accuracy[-1]), end="")
        sess.close()
    print("%.2f" % np.mean(accuracy[-10:]))
print("%.2f" % np.mean(accuracy))
