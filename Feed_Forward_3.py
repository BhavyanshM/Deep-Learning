import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_layer_1 = 500
n_layer_2 = 500
n_layer_3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# The Computation Graph
def neural_network_graph(data):

	# Just defining the weights and biases
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_layer_1])),
					  'biases':tf.Variable(tf.random_normal([n_layer_1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
					  'biases':tf.Variable(tf.random_normal([n_layer_2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_layer_2, n_layer_3])),
					  'biases':tf.Variable(tf.random_normal([n_layer_3]))}
	
	output_layer = {'weights':tf.Variable(tf.random_normal([n_layer_3, n_classes])),
					  'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	# Calculations of the layers
	layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	layer_2 = tf.nn.relu(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	layer_3 = tf.nn.relu(layer_3)

	output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

	return output
	
# Training the model with the input data for multiple epochs
def train_neural_network(x):
	prediction = neural_network_graph(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# At Default Learning rate
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_,c = sess.run([optimizer,cost], feed_dict = {x: epoch_x , y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

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