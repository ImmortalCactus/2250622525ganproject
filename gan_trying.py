import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

plt.ion()

batch_size = 256
g_dim = 128
training_label = 2
clip_range = 1000
keep_rate = 0.5

x_d = tf.placeholder(tf.float32, shape = [None, 784])
x_g = tf.placeholder(tf.float32, shape = [None, 128])
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

weights = {
    "w_d1" : weight_variable([784, 128]),
    "w_d2" : weight_variable([128, 1]),
    "w_g1" : weight_variable([128, 256]),
    "w_g2" : weight_variable([256, 784])
}

biases = {
    "b_d1" : bias_variable([128]),
    "b_d2" : bias_variable([1]),
    "b_g1" : bias_variable([256]),
    "b_g2" : bias_variable([784]),
}

var_d = [weights["w_d1"], weights["w_d2"], biases["b_d1"], biases["b_d2"]]
var_g = [weights["w_g1"], weights["w_g2"], biases["b_g1"], biases["b_g2"]]

def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    h_g2 = tf.add(tf.matmul(h_g1, weights["w_g2"]),biases["b_g2"])
    return h_g2


def discriminator(x):
    h_d1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(x, weights["w_d1"]), biases["b_d1"])),keep_rate)
    h_d2 = tf.add(tf.matmul(h_d1, weights["w_d2"]), biases["b_d2"])
    return h_d2

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

g_sample = generator(x_g)
d_real= discriminator(x_d)
d_fake = discriminator(g_sample)

d_loss = tf.reduce_mean(d_real) -  tf.reduce_mean(d_fake)
g_loss = -tf.reduce_mean(d_fake)
clip_D = [p.assign(tf.clip_by_value(p, -clip_range, clip_range))for p in var_d]
                         
d_optimizer = tf.train.RMSPropOptimizer(0.0005).minimize(-d_loss, var_list= var_d)
g_optimizer = tf.train.RMSPropOptimizer(0.0005).minimize(g_loss, var_list= var_g)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
for step in range(60001):
    for i in range(1):
        batch = mnist.train.next_batch(batch_size)
        counter = 0
        size=0
        for i in range(batch_size):
            if(batch[1][i][training_label]):
                size=size+1
        cleared_batch = np.ndarray(shape=(size,784))
        for i in range(batch_size):
            if(batch[1][i][training_label]):
                cleared_batch[counter]=batch[0][i]
                counter=counter+1
        
        d_loss_train = sess.run([d_optimizer, d_loss,clip_D], feed_dict = {x_d: cleared_batch, x_g: sample_Z(size, g_dim)})

    g_loss_train = sess.run([g_optimizer, g_loss], feed_dict = {x_g: sample_Z(size, g_dim)})

    if(step%500==0):
        pixels=sess.run(g_sample,feed_dict={x_g: sample_Z(1, g_dim)})
        pixels=pixels.reshape((28,28))
        plt.imshow(pixels,cmap="gray")
        plt.pause(0.001)
        plt.show()
