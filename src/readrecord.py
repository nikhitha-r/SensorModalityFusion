import tensorflow as tf
import os

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64)
      })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  label = tf.cast(features['label'], tf.int32)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  return image, label, height, width, depth

with tf.Session() as sess:
  filename_queue = tf.train.string_input_producer(["/Users/nikhitha/Documents/Practikum/training_0000/segment-1208303279778032257_1360_000_1380_000_with_camera_labels.tfrecord"])
  image, label, height, width, depth = read_and_decode(filename_queue)
  image = tf.reshape(image, tf.pack([height, width, 3]))
  image.set_shape([32,32,3])
  init_op = tf.initialize_all_variables()
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(1000):
    example, l = sess.run([image, label])
    print (example,l)
  coord.request_stop()
  coord.join(threads)
  
  