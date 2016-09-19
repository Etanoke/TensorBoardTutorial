import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn import cross_validation


def train(train_x, test_x, train_y, test_y):
    # softmax
    x = tf.placeholder(tf.float32, [None, 4], name='input')
    W = tf.Variable(tf.zeros([4, 3]), name='weight')
    b = tf.Variable(tf.zeros([3]), name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # training
    y_t = tf.placeholder(tf.float32, [None, 3], name='teacher_signal')
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_t * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(601):
            sess.run(train_step, feed_dict={x: train_x, y_t: train_y})
            if i % 10 == 0:
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_t, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                result = sess.run(accuracy, feed_dict={x: test_x, y_t: test_y})
                print('step: {:4d}: train accuracy: {:.4f}'.format(i + 1, result))


def prepare_dataset() -> (np.array, np.array):
    iris = learn.datasets.load_dataset('iris')
    # iris.target is 0,0,0...1,1,1...2,2,2...
    # convert to one-hot 150 * 3 teacher data
    one_hot_label = np.array([
        np.where(iris.target == 0, [1], [0]),
        np.where(iris.target == 1, [1], [0]),
        np.where(iris.target == 2, [1], [0])
    ]).T

    return iris.data, one_hot_label


def main():
    sample_inputs, sample_labels = prepare_dataset()
    # Cross Validation
    # use 80% data for training, and use 20% data for test
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(
        sample_inputs, sample_labels, test_size=0.2
    )
    train(train_x, test_x, train_y, test_y)

if __name__ == '__main__':
    main()
