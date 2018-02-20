from __future__ import division, print_function, unicode_literals


import os
import gym
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense

from PIL import Image

dims = (80,60)

for i in range(len(os.listdir('.'))-1):
	image = Image.open('canvas_image_{}.png'.format(i))
	image.thumbnail(dims)
	image = image.convert('L');
	image.save('../Images/canvas_image_{}.png'.format(i))





