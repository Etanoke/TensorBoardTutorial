from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/test', 'Summaries directory')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def le_net_conv_pool(input_image, filter_size, input_channel, output_channel):
    with tf.name_scope('Convolution'):
        W_conv = weight_variable([filter_size, filter_size, input_channel, output_channel])
        b_conv = bias_variable([output_channel])
        h_conv = tf.nn.relu(conv2d(input_image, W_conv) + b_conv)
    with tf.name_scope('MaxPool'):
        h_pool = max_pool_2x2(h_conv)
    return h_pool


def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    sess = tf.InteractiveSession()
    # Input Layer
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # First Convolution Layer
    with tf.name_scope('FirstLeNetConvPool'):
        input_image = tf.reshape(x, [-1, 28, 28, 1])
        out_image_l1 = le_net_conv_pool(input_image, 5, 1, 32)

    # Second Convolution Layer
    with tf.name_scope('SecondLeNetConvPool'):
        out_image_l2 = le_net_conv_pool(out_image_l1, 5, 32, 64)

    # Densely Connected Layer
    with tf.name_scope('DenselyConnectedLayer'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(out_image_l2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    with tf.name_scope('Dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    with tf.name_scope('ReadoutLayer'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Train the Model
    with tf.name_scope('Train'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        tf.scalar_summary('Cross Entropy', cross_entropy)

    # Evaluate the Model
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('Accuracy', accuracy)

    sess.run(tf.initialize_all_variables())

    # for TensorBoard, logging progress of training and test
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    merged_summaries = tf.merge_all_summaries()

    # Start Training!
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    train_log_interval = max(FLAGS.max_steps // 20, 1)
    test_log_interval = max(FLAGS.max_steps // 200, 1)
    for i in range(FLAGS.max_steps):
        batch = mnist.train.next_batch(50)
        # train step
        if i % train_log_interval == train_log_interval - 1:
            summary, _ = sess.run([merged_summaries, train_step],
                                  feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5},
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step{:05d}'.format(i))
            train_writer.add_summary(summary, i)
        else:
            summary, _ = sess.run([merged_summaries, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            train_writer.add_summary(summary, i)
        # test step
        if i % test_log_interval == test_log_interval - 1:
            summary, _ = sess.run([merged_summaries, accuracy],
                                  feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            test_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()


def main():
    # Delete old training log
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    train()


if __name__ == '__main__':
    main()
