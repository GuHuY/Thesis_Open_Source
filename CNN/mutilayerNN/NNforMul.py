from ty_lib import *
import random
from sklearn.model_selection import KFold



def compute_accuracy(v_data, v_lable):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_data})
    mse = tf.losses.mean_squared_error(y_pre, v_lable)
    loss = sess.run(mse)
    print([list(v_data[1]), list(v_lable[1]), list(y_pre[1])])
    return loss


data = np.load("MulData.npy")
random.shuffle(data)
print(data)
kf = KFold(n_splits=10)

accuracy = []
for train, test in kf.split(data):
    traning = np.array(data)[train]
    test = np.array(data)[test]
    training_data = traning[:, :-1].astype(np.float32)
    training_label =  np.array([[x] for x in traning[:, -1]]).astype(np.float32)
    vali_data = test[:, :-1].astype(np.float32)
    vali_label = np.array([[x] for x in test[:, -1]]).astype(np.float32)

    # x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
    # noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    # y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 4])
    ys = tf.placeholder(tf.float32, [None, 1])

    # hidden layers
    l1 = add_layer(xs, 4, 4)
    l2 = add_layer(l1, 4, 8)
    l3 = add_layer(l2, 4, 8)
    # l3 = add_layer(l2, 10,45)
    prediction = add_layer(l2, 8, 1) # output layer

    mse = tf.losses.mean_squared_error(prediction, ys)
    train_step = tf.train.AdamOptimizer(learning_rate=1).minimize(mse)
    # GradientDescentOptimizer(0.7).minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: training_data, ys: training_label})
    accuracy.append(compute_accuracy(vali_data, vali_label))
    print("fold "+str(len(accuracy))+"   Loss "+ "%.2f" % float(accuracy[-1]))
    sess.close()
print("%.2f" % np.mean(accuracy[-10:]))
