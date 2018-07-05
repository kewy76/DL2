# Kate Williams
# 7/5/2018

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# Improve the linear regression discussed in class with a new dataset (Smoking)

# Given model only has one variable "X"
# We need to consider two variables "X1" and "X2", two weights "W1" and "W2", and bias "b"
# Plot the linear regression of the form Y = X1*W1 + X2*W2+b (where X1 is smoking status and X2 is age classification)
# Show the graph on TensorBoard

DATA_FILE = 'Smoking.xls'  # Choose the correct file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")  # Open the file
sheet = book.sheet_by_index(0)  # Pick the correct sheet
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])  # Pick out array from sheet
n_samples = sheet.nrows - 1  # Take the number of values as a counter variable

# Plot the linear regression
# Basic linear regression code from in class source code
X1 = tf.placeholder(tf.float32, name='X1')  # Placeholder for X1 (Smoking status)
X2 = tf.placeholder(tf.float32, name='X2')  # Placeholder for X2 (Age classification)
Y = tf.placeholder(tf.float32, name='Y')  # Placeholder for Y
W1 = tf.Variable(0.0, name='W1')  # Create weights, initialized to 0
W2 = tf.Variable(0.0, name='W2')
b = tf.Variable(0.0, name='bias')  # Create bias, initialized to 0
Y_predicted = X1*W1 + X2*W2 + b  # Build model to predict Y
loss = tf.square(Y - Y_predicted, name='loss')  # Use the square error as the loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)  # Minimize loss
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize w and b
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for i in range(100):  # Train the model for 25 epochs
        total_loss = 0  # Set total loss to zero
        for x1, x2, x3, x4 in data:
            _, l = sess.run([optimizer, loss], feed_dict={X1: x1, X2: x2, Y: x4})  # Run session
            total_loss += l  # Fetch values of loss
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    writer.close()  # Close the writer
    W1, W2, b = sess.run([W1, W2, b])  # Output the values of w and b
    X1, X2 = data.T[0], data.T[1]  # Plot the results
    plt.plot(X1, 'bo', label='Real data')
    plt.plot(X2, 'bo', label='Real data')
    plt.plot(X1, X2, X1*W1+X2*W2+b, 'r', label='Predicted data')
    plt.legend()
    plt.show()
