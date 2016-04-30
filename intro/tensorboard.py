import numpy as np
import tensorflow as tf
x_data= np.random.rand(50,2).astype(float)
y_data = np.expand_dims(((x_data[:,1] > 0.5)*( x_data[:,0] > 0.5)).astype(int),axis=1)

x_ = tf.placeholder(tf.float32,[None,2], name="x_")
y_ = tf.placeholder(tf.float32,[None,1], name="y_")
w = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0), name="w")
b = tf.Variable(tf.zeros([1,1]), name="b")
y = tf.nn.sigmoid(tf.matmul(x_,w)+b)

with tf.name_scope("cross_entropy") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y) + (1-y_) * tf.log(1-y))

with tf.name_scope("trainer") as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cross_entropy)

with tf.name_scope("initializer") as scope:
    init = tf.initialize_all_variables()

tf.histogram_summary("weights", w)
tf.histogram_summary("biases", b)

tf.scalar_summary("cross_entropy", cross_entropy)
merged = tf.merge_all_summaries()
sess = tf.Session()
writer = tf.train.SummaryWriter("./", sess.graph_def)
sess.run(init)
for step in xrange(500):
    sess.run(train,feed_dict={x_:x_data,y_:y_data})
    if step % 20 == 0:
        print(step, sess.run(cross_entropy,feed_dict={x_:x_data,y_:y_data}))
        summary_str = sess.run(merged,feed_dict={x_:x_data,y_:y_data})
        writer.add_summary(summary_str, step)

