import random
import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)
tf.keras.utils.set_random_seed()

initializer = tf.keras.initializers.GlorotUniform()
initializer1 = tf.keras.initializers.GlorotUniform(0)

@tf.function
def fn():
   with tf.init_scope():
      for _ in range(5):
         print(initializer((4,)).numpy())

@tf.function
def fn1():
   with tf.init_scope():
      for _ in range(5):
         print(initializer1((4,)).numpy())

fn()
fn1()