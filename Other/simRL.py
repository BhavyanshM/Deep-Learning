from __future__ import division, print_function, unicode_literals

import gym
import random
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense

from PIL import Image

image = Image.open('../Images/canvas_image_{}.jpg'.format(0))
image.show()