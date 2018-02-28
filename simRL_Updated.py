from __future__ import division, print_function, unicode_literals


import os
import gym
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
from PIL import Image

batch_size = 128
output_classes = 10
epochs = 12
img_rows,img_cols = 28,28

dims = (80,60)

for i in range(len(os.listdir('.'))-1):
	image = Image.open('canvas_image_{}.png'.format(i))
	image.thumbnail(dims)
	image = image.convert('L');
	image.save('../Images/canvas_image_{}.png'.format(i))

# Loading Data
(x_train, y_train),(x_test, y_test) = mnist.load_data()


print(x_train.shape)
print(x_test.shape)

# Reshape hte data for consistency with Tensorflow conventions
if K.image_data_format == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

# Convert the type of data to float and normalize the pixel values from 0 to 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert the data into one-hot encoding
y_train = keras.utils.to_categorical(y_train, output_classes)
y_test = keras.utils.to_categorical(y_test, output_classes)

# Build the model using sequential stacking of layers such as Conv2D, MaxPooling2D, Dropout, Dense, and Flatten
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_classes, activation='softmax'))

# Setup the configuration of loss, optimizer, and output metrics for the model
model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adadelta(),
				metrics=['accuracy'])

# Fit the data into the model network with parameters: batch_size, epochs, verbose, validation_data (x_test, y_test)
model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			validation_data=(x_test, y_test))

# Evaluate the model using the validation data (x_test, y_test)
score = model.evaluate(x_test, y_test, verbose=0)

# Print score made by the model
print('loss:',score[0])
print('accuracy:',score[1])


# Attempt at the DQN starts here

# Perform the step to the the info about the reward and the next state, based on an action (See agent.act())
next_state, reward, done, info = env.step(action)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_classes, activation='softmax'))

# Once the model is built
model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

# Fit the data points consisting of state and rewards
model.fit(state, reward_value, epochs=1, verbose=1)

# Once trained the model can predict as follows
prediction = model.prediction(state)

# LOss = sq(r + gamma*maxQ(s, a') - Q(s,a))
target = reward + gamma * np.amax(model.predict(next_state))

# Function to store the previous state-action pairs
def remember(self, self, action, reward, next_state, done):
	self.memory.append((state, action, reward, next_state, done))

minibatch = random.sample(self.memory, batch_size)

for state, action, reward, next_state, done in minibatch:
	target = reward

	if not done:
		target = reward + self.gamma * np.amax(self.model.predict()[0])

	target_f = self.model.predict(state)
	target_f[0][action] = target

	self.model.fit(state, target_f, epochs=1, verbose=0)

# Function that the agent uses the choose the action
def act(self, state):
	if np.random.rand() <= self.epsilon:
		return env.action_space.sample()

	act_values = self.model.predict(state)

	return np.argmax(act_values[0])

# Code from the blog Keon made on using Keras and Gym
# Class named DQNAgent

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.994
		self.learning_rate = 0.001
		self.model = self.build_model()

	def build_model():
		model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
		model.add(Conv2D(64, (3,3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(output_classes, activation='softmax'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return env.action_space.sample()
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def remember(self, self, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
	env = gym.make()
	agent = DQNAgent(env)

	    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)












