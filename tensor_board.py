import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn import cross_validation

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('summaries_dir', '/tmp/tensor_logs', 'Summaries directory')

def train(train_x, test_x, train_y, test_y):
    sess = tf.InteractiveSession()

    # Create a 2 layer model, that have only input and output layer
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 4], name='x')
        y_t = tf.placeholder(tf.float32, [None, 3], name='y_teacher')

    with tf.name_scope('output'):
        W = tf.Variable(tf.zeros([4, 3]), name='weight')
        b = tf.Variable(tf.zeros([3]), name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_t * tf.log(y), reduction_indices=[1]))
        tf.scalar_summary('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    # for TensorBoard, logging progress of training and test
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
    tf.initialize_all_variables().run()
    merged_summaries = tf.merge_all_summaries()

    for i in range(FLAGS.max_steps):
        # train step
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged_summaries, train_step],
                                  feed_dict={x: train_x, y_t: train_y},
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step{:03d}'.format(i))
            train_writer.add_summary(summary, i)
        else:
            summary, _ = sess.run([merged_summaries, train_step], feed_dict={x: train_x, y_t: train_y})
            train_writer.add_summary(summary, i)
        # test step
        if i % 10 == 0:
            summary, test_accuracy = sess.run([merged_summaries, accuracy], feed_dict={x: test_x, y_t: test_y})
            test_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def prepare_iris_dataset() -> (np.array, np.array):
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
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # prepare sample from Iris Dataset
    sample_inputs, sample_labels = prepare_iris_dataset()
    # Cross Validation
    # use 80% data for training, and use 20% data for test
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(
        sample_inputs, sample_labels, test_size=0.2
    )
    train(train_x, test_x, train_y, test_y)

if __name__ == '__main__':
    main()
