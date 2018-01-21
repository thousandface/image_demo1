import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size=100

n_batch=mnist.train.num_examples // batch_size

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,W)+b)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init=tf.global_variables_initializer()

correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver=tf.train.Saver()  #保存模型

with tf.Session() as sess:
     sess.run(init)
     print( sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
     saver.restore(sess,'net/my_net.ckpt')
     print( sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
