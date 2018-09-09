import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_epochs = 10
n_classes = 10
batch_size = 100
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

# The Computation Graph
def recurrent_neural_network(x):

	# Just defining the weights and biases
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
			 'biases':tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)


	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output
	
# Training the model with the input data for multiple epochs
def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# At Default Learning rate
	optimizer = tf.train.AdamOptimizer().minimize(cost)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))


				_,c = sess.run([optimizer,cost], feed_dict = {x: epoch_x , y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)



# import numpy as np

# x = tf.constant([1,2,3])
# sess = tf.Session()
# tens1 = tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])
# print(tens1)
# print(sess.run(tens1)[1,1,1])
# npr = np.random.rand(32).astype(np.float32)
# y = tf.convert_to_tensor(npr, dtype=tf.float32)
# print(npr)
# print(sess.run(y))
# r = tf.constant(npr)
# print(sess.run(r))