# Kate Williams
# 7/5/2018

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# Plot the linear regression for the USA_Housing dataset

# Change the dataset from "Fire_Theft" to "USA_Housing"
DATA_FILE = 'USA_Housing.xls.xlsx'  # Choose the correct file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")  # Open the file
sheet = book.sheet_by_index(0)  # Pick the correct sheet
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])  # Pick out array from sheet
n_samples = sheet.nrows - 1  # Take the number of values as a counter variable

# Plot the linear regression for average area house age vs. price
# Linear regression code from in class source code
X = tf.placeholder(tf.float32, name='X')  # Placeholder for X (average area house age)
Y = tf.placeholder(tf.float32, name='Y')  # Placeholder for Y (price)
w = tf.Variable(0.0, name='weights')  # Create weight, initialized to 0
b = tf.Variable(0.0, name='bias')  # Create bias, initialized to 0
Y_predicted = X * w + b  # Build model to predict Y
loss = tf.square(Y - Y_predicted, name='loss')  # Use the square error as the loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)  # Minimize loss
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize w and b
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for i in range(25):  # Train the model for 25 epochs
        total_loss = 0  # Set total loss to zero
        for x1, x2, x3, x4, x5, x6 in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x2, Y: x6})  # Run session
            total_loss += l  # Fetch values of loss
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    writer.close()  # Close the writer
    w, b = sess.run([w, b])  # Output the values of w and b
    X, Y = data.T[0], data.T[1]  # Plot the results
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, X*w+b, 'r', label='Predicted data')
    plt.legend()
    plt.show()

# Plot the linear regression for average area number of rooms vs. price
# Linear regression code from in class source code
X = tf.placeholder(tf.float32, name='X')  # Placeholder for X (average area house age)
Y = tf.placeholder(tf.float32, name='Y')  # Placeholder for Y (price)
w = tf.Variable(0.0, name='weights')  # Create weight, initialized to 0
b = tf.Variable(0.0, name='bias')  # Create bias, initialized to 0
Y_predicted = X * w + b  # Build model to predict Y
loss = tf.square(Y - Y_predicted, name='loss')  # Use the square error as the loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)  # Minimize loss
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize w and b
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for i in range(25):  # Train the model for 25 epochs
        total_loss = 0  # Set total loss to zero
        for x1, x2, x3, x4, x5, x6 in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x3, Y: x6})  # Run session
            total_loss += l  # Fetch values of loss
        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    writer.close()  # Close the writer
    w, b = sess.run([w, b])  # Output the values of w and b
    X, Y = data.T[0], data.T[1]  # Plot the results
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, X*w+b, 'r', label='Predicted data')
    plt.legend()
    plt.show()

# Show the graph on TensorBoard
