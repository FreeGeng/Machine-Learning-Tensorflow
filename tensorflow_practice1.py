# This practice is to train the network to know y = 0.1*x+0.3
# We give the network x and y and let it find the parameters 0.1 and 0.3
# We call the position of 0.1 "weight"(權重) and the position of 0.3 "bias"(偏差)

# numpy is used for data anaysis
import tensorflow as tf
import numpy as np

# create 100 random number data with np original type float32
x = np.random.rand(100).astype(np.float32)
y = x*0.1+0.3

###   create tensorflow structure start   ###
# use random_uniform to create a random array of numbers which values are between -1.0 and 1.0
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# the initial number of bias is 0, we want it to learn the truly bias step by step through the input data
Biases = tf.Variable(tf.zeros([1]))

y_2 = Weights*x + Biases

# to calculate the difference between the y_2(the y we trained) and the correct y
# obviously, in the beginning, the loss will be large , however the loss will be smaller during the training
loss = tf.reduce_mean(tf.square(y_2-y))

# the two lines below is to tell the network to create a optimizer to decrease the loss during EVERY TRAINING
# So we can let the network better and better
# There is a lot of optimizer, GradientDescent is the simplest
# 0.5 is call learning efficient(學習效率), it must < 1 , we use 0.5 for practice
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# after create the structure, we need to activate the structure and create the initial numbers
init = tf.global_variables_initializer()
###   create tensorflow structure end   ###


#run our structure, vert important step!
sess = tf.Session()
sess.run(init)

# let the network be trained 10001 times
# every 500 steps, we print the current Weights and Biases for observation
for step in range(501):
    sess.run(train)
    if step %20 == 0:
        print(step,sess.run(Weights),sess.run(Biases))