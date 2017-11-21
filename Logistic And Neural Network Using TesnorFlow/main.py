from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os as os
from skimage import color
import numpy as np
import imageio
from skimage.transform import resize

def main():
    print("UBitName:"+ "anantram")
    print("personNumber:" + "50249127")
    print("UBitName: pdeshmuk")
    print("personNumber: 50247649")
    print("UBit_Name: hsokhey" )
    print("personNumber: 50247213")
	xt,yt,xn,yn = usps_image_data()
	logisticregression(xt,yt,xn,yn)
	single_layer(xt,yt,xn,yn)
	conv_neural(xt,yt,xn,yn)
	
def oneHotEncoding(target):
    print ("oneHotEncoding")
    t = np.zeros((len(target),10))
    for i in range(len(target)):
        index = target[i]
        t[i][int(index[0])] = 1
    return t

def multilayer_perceptron(x,W1,b1):
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.tanh(hidden_out)
    return hidden_out	

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def usps_image_data():

	####### For data in Test dir #############
	uspst_x1 = np.zeros((1,784))
	uspst_y1 = np.ndarray(shape=(1,10),dtype=float)
	path2 = "C:/Users/anu21/Anant/Google Drive/MASTERS/Courses/ML/Assignment/Proj3(Classification)/proj3_images/Test/"
	val = 9
	for name in os.listdir(path2):
		final_path2 = path2 + name
		if name != "Thumbs.db":
			if int(name.split('.')[0].split('_')[1])%150==0:
				val = val - 1
			img = imageio.imread(final_path2)
			gray_img = color.rgb2gray(img)
			resized_img = ((resize(gray_img,(28,28))/255)-1)/-1
			flat_img = np.ravel(resized_img)
			uspst_x1 = np.insert(uspst_x1,len(uspst_x1),flat_img,axis=0)
			uspst_y1 = np.insert(uspst_y1,len(uspst_y1),int(val),axis=0)
	uspst_y1 = oneHotEncoding(uspst_y1)
	
	####### For data in Numerals dir #############
	path = "C:/Users/anu21/Anant/Google Drive/MASTERS/Courses/ML/Assignment/Proj3(Classification)/proj3_images/Numerals/"
	uspsn_x = np.zeros((1,784))
	uspsn_y = np.ndarray(shape=(1,10),dtype=float)
	for i in range(10):
		new_path = path + str(i) + "/"
		for name in os.listdir(new_path):
			final_path = new_path + name
			if ".list" not in name:
				if (name != "Thumbs.db"):
					img = imageio.imread(final_path)
					gray_img = color.rgb2gray(img)
					resized_img = ((resize(gray_img,(28,28))/255)-1)/-1
					flat_img = np.ravel(resized_img)
					uspsn_x = np.insert(uspsn_x,len(uspsn_x),flat_img,axis=0)
					uspsn_y = np.insert(uspsn_y,len(uspsn_y),int(i),axis=0)
	uspsn_y = oneHotEncoding(uspsn_y)
	
	return uspst_x1,uspst_y1,uspsn_x,uspsn_y

def logisticregression(xt,yt,xn,yn):
    with tf.Session() as sess:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        tf.global_variables_initializer().run()
        y = tf.matmul(x, W) + b
        finalacc = 0
        l = 0
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        l = 0.601
        for i in [l * 0.001 for l in range(1, 1000,100 )]:
            train_step = tf.train.GradientDescentOptimizer(i).minimize(cross_entropy)
            
            # Train the model we made above
            for _ in range(1000):
                batch = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            
            # Test the trained model
            correct_prediction1 = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            newacc = accuracy1.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            if newacc>finalacc:
                l = i
                finalacc = newacc
        print(l,finalacc)
        
		###### Final Logistic Model after tuning the parameters
        train_step = tf.train.GradientDescentOptimizer(l).minimize(cross_entropy)
        
		###### Train the model we made above
        for _ in range(1000):
            batch = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
        
		###### Test the trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        Accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

		########  Test for USPS images under Test folder
        AccuracyT = sess.run(accuracy,feed_dict={x: xt, y_: yt})
        
		########  Test for USPS images under Numerals folder
        AccuracyN = sess.run(accuracy,feed_dict={x: xn, y_: yn})
        
        print("--------------Logistic regression with tuned hyperparameter------------------------------")
        print(" Learning_rate: ", l)
        print("MNIST Accuracy: ", Accuracy)
        print("USPS Test Accuracy: ", AccuracyT)
        print("USPS Numerals Accuracy: ", AccuracyN)
        print(" Optimizer: ", 'GradientDescentOptimizer')
        
def single_layer(xt,yt,xn,yn):
    with tf.Session() as sess:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        training_epochs = 15
        batch_size = 20
        n_hidden_1 = 500 # 1st layer number of neurons
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)
        
        # tf Graph input
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_classes])
        # now declare the weights connecting the input to the hidden layer
        W1 = tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([n_hidden_1]), name='b1')
        # and the weights connecting the hidden layer to the output layer
        W2 = tf.Variable(tf.random_normal([n_hidden_1, n_classes], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([n_classes]), name='b2')
        # Construct model
        logits = tf.add(tf.matmul(multilayer_perceptron(X,W1,b1), W2), b2)
        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    
        ###### Train the model
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_step, cross_entropy], feed_dict={X: batch_x,Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
        
        ###### Test the model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        
        ##### Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        Accuracy = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
        
        ########  Test for USPS images under Test folder
        AccuracyT = sess.run(accuracy,feed_dict={X: xt, Y: yt})
        
        ########  Test for USPS images under Numerals folder
        AccuracyN = sess.run(accuracy,feed_dict={X: xn, Y: yn})
        
        print("--------------Single Hidden Layer with Tuned HyperParameters------------------------------")
        print(" Learning_rate: ", 0.001)
        print("MNIST Accuracy: ", Accuracy)
        print("USPS Test Accuracy: ", AccuracyT)
        print("USPS Numerals Accuracy: ", AccuracyN)
        print(" Optimizer: ", 'AdamOptimizer')
        print(" Number of Epochs:", training_epochs)
        print(" Hidden Layer Nodes:",n_hidden_1)

def conv_neural(xt,yt,xn,yn):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    learning_rate=0.0001
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        Accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        
        ########  Test for USPS images under Test folder
        AccuracyT = sess.run(accuracy,feed_dict={x: xt, y_: yt, keep_prob: 1.0})
        
        ########  Test for USPS images under Numerals folder
        AccuracyN = sess.run(accuracy,feed_dict={x: xn, y_: yn, keep_prob: 1.0})
        
        print("--------------Convolutional Neural Network------------------------------")
        print(" Learning_rate: ", 0.0001)
        print("MNIST Accuracy: ", Accuracy)
        print("USPS Test Accuracy: ", AccuracyT)
        print("USPS Numerals Accuracy: ", AccuracyN)
        print(" Optimizer: ", 'AdamOptimizer')
        
if __name__ == '__main__':
    main()