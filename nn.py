import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)
# X is our image, and the specs of the image Pixel 28x28 and 1 for greyscale-
# None represents the number of images
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
#Weight
W = tf.Variable(tf.zeros([784, 10]))
#Bias
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)
