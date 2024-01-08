# Check Tensorflow can use GPUs
import tensorflow as tf
tf.test.is_gpu_available()