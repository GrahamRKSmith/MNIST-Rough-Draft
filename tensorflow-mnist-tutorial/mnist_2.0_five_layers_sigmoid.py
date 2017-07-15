# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
#GS: This is for debugging purposes mainly, so that we can reproduce results
#GS: Each time we compute a mini-batch, it will be the same
tf.set_random_seed(0)

# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
#GS: This gets the data from tensorflow.
#GS: The validation_size = 0 means that we don’t want a validation set - all the data should be in the training set
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
#GS: The X is a placeholder - this is going to be the matrix of all the tensor of image files
#GS: 28*28 is the size of the image, None is used so that we can feed it any batch size, and the 1 is the number of channels
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
#GS: used to store the values of the labels that correspond to each image
Y_ = tf.placeholder(tf.float32, [None, 10])

# five layers and their number of neurons (tha last layer has 10 softmax neurons)
#GS: a label for the number of neurons per layer
#GS: We start with 200 neutrons in the first layer, and progressively make the network smaller
L = 200
M = 100
N = 60
O = 30
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
#GS: Each number represents a layer; W1 and B1 are the weights and biases for the first layer
#GS: tf.truncated_normal([784, L], stddev=0.1) gives a matrix of weights for 784 features and 300 neurons
#GS: truncated normal is like a normal distribution except the tails are chopped off
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# The model
#GS: our original data has a shape of (batch_size, 28, 28, 1)
#GS: XX = tf.reshape(X, [-1, 784]) is flattening the 28 * 28 part into a vector
#GS: so we have a matrix of batch_size, 784 to feed into the first layer
#GS: Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1) this line takes this matrix XX,
#GS: multiplies it by the weights W1 which so far have been randomly initialized,
#GS: and adds the bias weights B1, which are zeros.
#GS: XX would have shape (100, 784) if batch size was 100
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
#GS: this is Y1, and fed into the next one
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
#GS: at the end softmax is used instead of sigmoid
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
#GS: This is using tensor flow’s library implementation of cross entropy which is numerically stable
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
#GS: computing the average of the loss function across all the examples in the batch
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
#GS: This compares all the predictions with the actual labels
#GS: the argmax takes the index of the maximum probability
#GS: so for example, if you have this vector: [0, 0, 0, 0.6, 0.3, 0.1], argmax would be 3
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
#GS: the correct_predictions variable now contains a vector of values which are ones and zeroes
#GS: ones representing a correct prediction
#GS: reduce_mean here computes the percentage of correct values
#GS: It has to be cast to a float because the values of correct_predictions are ints
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
#GS: visualization again; It’s creating plots to chart the accuracy, weights, loss, biases
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# training step, learning rate = 0.003
#GS: this is a hyper parameter for the optimizer (in this case Adam)
#GS: Adam is like GradientDescent but it has a few extra optimizations
#GS: one of them is learning rate decay
#GS: learning rate decay means that the learning rate will start out high,
#GS: and slowly decrease (exponentially) over time during training
learning_rate = 0.003
#GS: calls the Adam optimizer
#GS: we have to specify what it is we want to minimize (in this case cross_entropy).
#GS: In some cases, you can have a different loss function or even multiple loss functions (depending on your goals)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# init
#GS: tf.global_variables_initializer() creates the init node in the computational graph
#GS: It doesn’t actually initialize the variables until sess.run(init) is executed
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
#GS: this function executes the training steps and performs the animations
#GS: the variable i indicates the number of total steps you want to run
#GS: update_test/train_data are boolean values which will either do training or testing
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    #GS: This grabs a mini-batch of size 100 randomly from the data set
    #GS: It is far more efficient to perform gradient descent on a mini-batch instead of the entire data set
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    #GS: That part runs if we want to perform training, it is mainly just the visualizer
    #GS: we also store the variables for accuracy, entropy, weights, biases, etc.
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], {X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    #GS: same thing as above, the  str(i*100//mnist.train.images.shape[0]+1)  part computes the epoch number
    #GS: but this is for the test set
    #GS: Running the model on the test set allows us to evaluate how well the model is generalizing to unseen data
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], {X: mnist.test.images, Y_: mnist.test.labels})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    #GS: That performs the main computation, which is to compute a step of gradient descent
    #GS: gradient descent is an algorithm to optimize a loss function
    #GS: backprop is how the earlier neutrons in the network figure out what their error is relative to the loss function
    #GS: backprop uses the chain rule to make a backward pass
    #GS: once backprop is done, gradient descent knows which direction to go in to reduce the loss
    sess.run(train_step, {X: batch_X, Y_: batch_Y})

#GS: This runs the entire program - datavis is an object of the MnistDataVis() class
#GS: It runs the animate method, which in turn runs everything else in the program
datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

#GS: prints the accuracy of the best run to the console
print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# Some results to expect:
# (In all runs, if sigmoids are used, all biases are initialised at 0, if RELUs are used,
# all biases are initialised at 0.1 apart from the last one which is initialised at 0.)

## learning rate = 0.003, 10K iterations
# final test accuracy = 0.9788 (sigmoid - slow start, training cross-entropy not stabilised in the end)
# final test accuracy = 0.9825 (relu - above 0.97 in the first 1500 iterations but noisy curves)

## now with learning rate = 0.0001, 10K iterations
# final test accuracy = 0.9722 (relu - slow but smooth curve, would have gone higher in 20K iterations)

## decaying learning rate from 0.003 to 0.0001 decay_speed 2000, 10K iterations
# final test accuracy = 0.9746 (sigmoid - training cross-entropy not stabilised)
# final test accuracy = 0.9824 (relu - training set fully learned, test accuracy stable)
