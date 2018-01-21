import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minist=input_data.read_data_sets('MNIST_data',one_hot=True)

#输入图片是28*28
n_input=28
max_time=28
lstm_size=100
n_classes=10
batch_size=50
n_batch=minist.num_examples // batch_size

#
x=tf.placeholder(tf.float32,[None,784])
#标签
y=tf.placeholder(tf.float32,[None,10])
#初始化权值
weights=tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0.1))
#初始化偏置值
biases=tf.Variable(tf.concat(0.1,shape=[n_classes]))

#定义RNN网络
def RNN_1(X,weights,biases):
    inputs=tf.reshape(X,[-1,max_time,n_input])
    #定义LSTM基本CELL
    lstm_cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
     # final_state[0]是cell state
    # final_state[1]是hidden_state                                          #传28次每一次传28，
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results

#计算RNN的返回结果
prediction=RNN_1(x,weights,biases)
#损失函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用adam优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
#求准确几率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys=minist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc=sess.run(accuracy,feed_dict={x:minist.test.images,y:minist.test.images})
        print('Iter'+str(epoch)+'testing acc='+str(acc))
