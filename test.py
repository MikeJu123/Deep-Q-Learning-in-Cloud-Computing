import numpy as np
import tensorflow as tf

# x = tf.constant([[1, 4],[5,6],[11,15],[323,5345]])
# y = tf.constant([2, 5])
# z = tf.constant([3, 6])

# x2 = np.array([1,2])
# y2 = np.array([[3,4],[5,6]])
# z2 = np.array([5,6])
# c = tf.unstack(x,axis = 1)  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
# d = tf.shape(x)

# # with tf.Session() as sess:
# #     print (sess.run(d))
# c = np.ones([5,1])
# print(c)

input = tf.ones([5, 1])  # [5,1]的矩阵，5组数据，每组数据为1个。tf.layers.dense会根据这个shape，自动调整输入层单元数。
output = tf.layers.dense(input, 10)
print(output.get_shape())

input = tf.ones([3, 2])
output = tf.layers.dense(input, 10)
print(output.get_shape())

input = tf.ones([1, 7, 20])
output = tf.layers.dense(input, 10)
print(output.get_shape())

input = tf.ones([1, 7, 11, 20])
output = tf.layers.dense(input, 10)
print(output.get_shape())
