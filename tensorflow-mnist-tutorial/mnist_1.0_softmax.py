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

#GS: Tensorflow is the machine learning library most used today
#GS: MNIST is the largest classification of handwritten digits at 60,000 training
#GS: and 10,000 testing
#GS: the tensorflowvisu is a py file that interfaces with matplotlib for visualisations
#GS: this sets a random seed for tensorflow; this is to get stable results in randomization each time
import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
#GS: one-hot encoding means each digit is represented as an array of 10 values all 0s and one 1
#GS: 'validation_size = 0' = for a given digit, the system would compute probabilities for it being one of the digits
#GS: from 0 to 9 and go with the one with the highest probability
#GS: the below is calling the function 'read_data_sets' which is called from the 'mnist_data'
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
#GS: placeholder is literally the placeholder for each digit image (28 x 28 pixels)
#GS: the first argument is the data type (tf.float32) and the second is shape
#GS: None is for indexing the images in a minibatch
#GS: X and Y are placeholders that tensorflow will use for processing each input and output
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
#GS: placeholder for output digits 0-9
#GS: The 10 number is the number of results 0-9, the correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
#GS: this is for storing each digit as a series of 784 bits
#GS: each 'neuron' does a weighted sum of all of its inputs and feeds the result
#GS: through some non-linear activation function
#GS: weights are the intensities of each pixel; a digit is made up of 28x28 pixels
#GS: so we make it as a single array of 784 values; each has a weight based on a pixel intensity
#GS: intensity = lighter or darker
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
#GS: bias is a constant we add to each neuron
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
#GS: its easier to work with a single array than a 2d array
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
# GS: matmul is matrix multiply; the 'xx' is the input image represented in a single line of 784 values
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
#GS: cross entropy is the loss function we are trying to minimize.
#GS: reduce_mean is a tensorflow function that calculates cross-entropy for us
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0  # normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
#GS: now calculate the accuracy - it is similar to the calculation of cross-entropy
#GS: in tf.argmax(Y, 1) the Y is the input here, and 1 is the axis on which the input
#GS: should be processed. Argmax will use the input tensor Y and returns the index of
#GS: the largest value. Here Y is: Y = tf.nn.softmax(tf.matmul(XX, W) + b)
#GS: the variable "correct_prediction" has actual and predicted models (Y_ has the correct values)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
#GS: the below measures the accuracy on what we predicted.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
#GS: there are many optimizers you can use - here we are using the gradient descent optimizer
#GS: this is where cross-entropy will be minimzed (the actual process)
#GS: the .005 is the learning rate
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualisation - in other cases you could use TensorBoard
# TensorBoard is the visualization tool built for tensorflow
# TensorBoard allows you to follow your distributed TensorFlow job on remote areas
#GS: we are flattening the weights matrix. The learning rate is the size of each step.
allweights = tf.reshape(W, [-1])
#GS: this is the same as flattening weights, but we do it for biases matrix
allbiases = tf.reshape(b, [-1])
#GS: This is for all the fancy animations found in tensorflowvisu.py
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
#GS: This creates an init node in the computational graph - it's usually the
#GS: last step before starting a session
#GS: Each variable in tensorflow needs to be initialized,
#GS: and the global_variables_initializer initializes everything simultaneously
#GS: Instead of having to initialize each variable individually
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
#GS: this function mainly contains the code to run the animations and train the model
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    #GS: The first one extracts a batch of 100 examples from the data
    #GS: - this is chosen randomly from the 50,000 or so examples in the dataset
    #GS: This is known as a mini-batch - it is used with stochastic gradient descent
    #GS: in order to make the computation more efficient (rather than computing gradients
    #GS: for every single example in the entire data set)
    #GS: A good exercise is to try to compute gradients for the entire data set and see the performance difference.
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    #GS: This is for the visualization when the model is training on data
    if update_train_data:
        #GS: Notice how sess.run() has a list of variables as the first argument
        #GS: When you pass variables as a list, sess.run() will compute them once.
        #GS: If you don’t do this, it will recompute earlier variables each time.
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        #GS: That is printing the accuracy and loss to the console
        #GS: To see the numbers in addition to the graphs
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        #GS: The str(i*100) is computing the epoch so the i has 20001 iterations
        #GS: It is multiplied by 100 and then divided by the size of the dataset (i.e. mnist.train.images.shape[0])
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    #GS: Backprop is the way the error is propagated from the cost function to the earlier layers of the network
    #GS: This is how gradient descent figures out which direction to take a step in
    #GS: this is the part that actually does all the work - feed_dict is used to
    #GS: place the actual data into the placeholder variables (X, and Y_) which we
    #GS: defined when constructing the graph, and the train step operation does one step of GradientDescent
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

#GS: That is calling the animate method on the datavis object we created - this is related to that tensorflowvis file again
#GS: It runs the whole program
datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

#GS: prints the text accuracy - this is computed by figuring out the percentage of images that are classified correctly
print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
