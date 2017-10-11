# encoding:utf-8
import sys

sys.path.append("..")  # 将该目录加入到环境变量

import pickle


def save_data(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))
    pass


def load_data(filename):
    f = open(filename, 'rb')
    model = pickle.load(f)
    print('load data model finished.')
    return model


def user_url_to_mf(user_nrl):
    D_input = 30
    D_hidden = 20
    D_label = 15

    x = tf.placeholder(tf.float32, [None, D_input], name="x")
    y = tf.placeholder(tf.float32, [None, D_label], name="y")

    W_h1 = tf.Variable(tf.truncated_normal([D_input, D_hidden], stddev=0.1), name="W_h")
    b_h1 = tf.Variable(tf.constant(0.0, shape=[D_hidden]), name="b_h")
    pre_act_h1 = tf.matmul(x, W_h1) + b_h1
    act_h1 = tf.nn.relu(pre_act_h1, name='act_h')

    W_o = tf.Variable(tf.truncated_normal([D_hidden, D_label], stddev=0.1), name="W_o")
    b_o = tf.Variable(tf.constant(0.0, shape=[D_label]), name="b_o")
    pre_act_o = tf.matmul(act_h1, W_o) + b_o
    p = tf.nn.relu(pre_act_o, name='act_y')

    with tf.Session() as sess:
        saver.restore(sess, "../data/fnn/my_net/fnn.ckpt")

        test_x = np.array([0.44837999, -2.91398406, -2.70611191, 1.69465101, 1.05228806,
                           0.28670901, 1.099159, -0.95564801, -4.48727512, -0.79399401,
                           -1.76026201, 3.64363003, 0.50814402, -2.068964, 3.65591908,
                           4.69985294, 2.428092, 0.87958002, 1.076895, -2.62675691,
                           -3.32081103, -1.05760396, 0.76970202, -2.30422211, -0.092633,
                           5.41511011, 3.06955695, -4.61815023, 0.49307701, 1.81028497])

        print(sess.run(p, feed_dict={x: test_x}))
